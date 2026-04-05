import torch
from .config import ModelConfig


class AshareBacktest:
    """A股截面选股回测引擎。

    每个交易日对所有股票按 alpha 分值排序，做多排名前 N 只。
    已内置 T+1 规则（通过 open-to-open target_ret 实现）、
    佣金+印花税、涨跌停过滤、换手率过滤。
    """

    def __init__(self):
        self.buy_cost = ModelConfig.TOTAL_BUY_COST
        self.sell_cost = ModelConfig.TOTAL_SELL_COST
        self.min_turnover = ModelConfig.MIN_TURNOVER_RATE
        self.top_n = ModelConfig.TOP_N_STOCKS

    def evaluate(self, factors, raw_data, target_ret):
        """
        Args:
            factors:    [num_stocks, T] alpha 信号（来自 PrefixVM）
            raw_data:   dict，含 'turnover_rate', 'pct_chg'（如有）, 'vol', 'close'
            target_ret: [num_stocks, T] open-to-open 前向收益
        Returns:
            (fitness: scalar tensor, avg_daily_return: float)
        """
        turnover_rate = raw_data.get("turnover_rate",
                            torch.zeros_like(factors))

        # 只在训练期评估
        T = factors.shape[1]
        split = int(T * ModelConfig.TRAIN_RATIO)
        factors = factors[:, :split]
        target_ret = target_ret[:, :split]
        turnover_rate = turnover_rate[:, :split]

        N_stocks, T_len = factors.shape

        # 排除停牌股票（NaN 信号或 NaN target_ret）
        valid_mask = ~(torch.isnan(factors) | torch.isnan(target_ret))
        scores = factors.clone()
        scores[~valid_mask] = float('-inf')
        scores[turnover_rate <= self.min_turnover] = float('-inf')

        # 市场状态判断：等权组合收益的 20 日均线（排除停牌）
        valid_target = target_ret.clone()
        valid_target[~valid_mask] = 0.0
        valid_count = valid_mask.float().sum(dim=0).clamp(min=1)
        market_daily_ret = valid_target.sum(dim=0) / valid_count
        market_ma20 = self._rolling_mean_1d(market_daily_ret, 20)

        # 一次性对所有交易日取 topk
        _, topk_idx = scores.topk(self.top_n, dim=0)   # [top_n, T_len]
        position = torch.zeros_like(factors)
        position.scatter_(0, topk_idx, 1.0)
        # 持仓中排除无效股票
        position[~valid_mask] = 0.0

        # 熊市减仓：市场弱势时持仓权重减半
        bear_mask = market_ma20 < 0
        position[:, bear_mask] *= 0.5

        # 换手
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0.0
        turnover = torch.abs(position - prev_pos)

        # 交易成本
        avg_cost = (self.buy_cost + self.sell_cost) / 2.0
        tx_cost = turnover * avg_cost

        # PnL（绝对收益）
        gross_pnl = position * target_ret
        net_pnl = gross_pnl - tx_cost

        # 每天 portfolio 收益（持仓股票的平均收益）
        daily_pnl = net_pnl.sum(dim=0) / (self.top_n + 1e-9)

        # ---- Fitness 计算 ----
        # 1) Sortino 比率作为主指标
        mean_ret = daily_pnl.mean()
        neg_returns = daily_pnl[daily_pnl < 0]
        if len(neg_returns) > 5:
            downside_std = neg_returns.std()
        else:
            downside_std = daily_pnl.std() + 1e-6

        sortino = mean_ret / (downside_std + 1e-6)

        # 2) 惩罚项
        cum_ret = daily_pnl.sum()

        # 最大回撤惩罚
        cum_curve = torch.cumsum(daily_pnl, dim=0)
        running_max = torch.cummax(cum_curve, dim=0)[0]
        drawdown = cum_curve - running_max
        max_dd = drawdown.min()
        dd_penalty = torch.relu(-max_dd - 0.05) * 2.0

        # 换手率惩罚
        avg_turnover = turnover.mean()
        turnover_penalty = 0.0
        if avg_turnover > 0.5:
            turnover_penalty = 1.0

        # 活跃度不足惩罚
        active_days = (position.sum(dim=0) > 0).float().sum()
        if active_days < 10:
            return torch.tensor(-10.0, device=factors.device), 0.0

        # 负收益惩罚
        neg_return_penalty = 0.0
        if cum_ret < 0:
            neg_return_penalty = min(abs(cum_ret) * 2.0, 1.0)

        # IC 奖励：截面预测能力（排除停牌日）
        ic_bonus = 0.0
        if N_stocks > 10:
            # 将 NaN 替换为 0 做排名
            f_clean = torch.nan_to_num(factors[:, :T_len], nan=0.0)
            alpha_rank = f_clean.argsort(dim=0).argsort(0).float()
            ret_clean = torch.nan_to_num(target_ret, nan=0.0)
            ret_rank = ret_clean.argsort(dim=0).argsort(0).float()
            ic_per_day = (alpha_rank * ret_rank).sum(dim=0) / (
                alpha_rank.norm(dim=0) * ret_rank.norm(dim=0) + 1e-6
            )
            ic_bonus = ic_per_day.mean() * 5.0

        # 综合得分
        fitness = sortino - dd_penalty - turnover_penalty - neg_return_penalty + ic_bonus

        return fitness, cum_ret.item()

    def evaluate_oos(self, factors, raw_data, target_ret):
        """在 OOS 区间评估 fitness，用于训练时的 OOS 验证。
        逻辑与 evaluate 一致，但使用后半段数据。"""
        turnover_rate = raw_data.get("turnover_rate",
                            torch.zeros_like(factors))

        T = factors.shape[1]
        split = int(T * ModelConfig.TRAIN_RATIO)
        factors = factors[:, split:]
        target_ret = target_ret[:, split:]
        turnover_rate = turnover_rate[:, split:]

        N_stocks, T_len = factors.shape
        if T_len < 20:
            return torch.tensor(0.0, device=factors.device)

        valid_mask = ~(torch.isnan(factors) | torch.isnan(target_ret))
        scores = factors.clone()
        scores[~valid_mask] = float('-inf')
        scores[turnover_rate <= self.min_turnover] = float('-inf')

        valid_target = target_ret.clone()
        valid_target[~valid_mask] = 0.0
        valid_count = valid_mask.float().sum(dim=0).clamp(min=1)
        market_daily_ret = valid_target.sum(dim=0) / valid_count
        market_ma20 = self._rolling_mean_1d(market_daily_ret, 20)

        _, topk_idx = scores.topk(self.top_n, dim=0)
        position = torch.zeros_like(factors)
        position.scatter_(0, topk_idx, 1.0)
        position[~valid_mask] = 0.0

        bear_mask = market_ma20 < 0
        position[:, bear_mask] *= 0.5

        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0.0
        turnover = torch.abs(position - prev_pos)
        avg_cost = (self.buy_cost + self.sell_cost) / 2.0
        tx_cost = turnover * avg_cost

        gross_pnl = position * target_ret
        net_pnl = gross_pnl - tx_cost
        daily_pnl = net_pnl.sum(dim=0) / (self.top_n + 1e-9)

        mean_ret = daily_pnl.mean()
        neg_returns = daily_pnl[daily_pnl < 0]
        if len(neg_returns) > 5:
            downside_std = neg_returns.std()
        else:
            downside_std = daily_pnl.std() + 1e-6
        sortino = mean_ret / (downside_std + 1e-6)

        cum_ret = daily_pnl.sum()
        cum_curve = torch.cumsum(daily_pnl, dim=0)
        running_max = torch.cummax(cum_curve, dim=0)[0]
        drawdown = cum_curve - running_max
        max_dd = drawdown.min()
        dd_penalty = torch.relu(-max_dd - 0.05) * 2.0

        neg_return_penalty = 0.0
        if cum_ret < 0:
            neg_return_penalty = min(abs(cum_ret) * 2.0, 1.0)

        oos_fitness = sortino - dd_penalty - neg_return_penalty
        return oos_fitness

    @staticmethod
    def _rolling_mean_1d(x, window):
        """对 1D tensor 计算滚动均值。"""
        T = x.shape[0]
        if T < window:
            return torch.zeros_like(x)
        pad = torch.zeros(window - 1, device=x.device)
        x_pad = torch.cat([pad, x])
        windows = x_pad.unfold(0, window, 1)
        return windows.mean(dim=-1)


class SingleStockBacktest:
    """单股票时序回测（兼容 times.py 模式，用于单标的验证）。"""

    def __init__(self, cost_rate=0.0005):
        self.cost_rate = cost_rate

    def evaluate(self, factors, target_ret):
        """
        Args:
            factors:    [T] 1D alpha 信号
            target_ret: [T] 1D 前向收益
        Returns:
            sortino: scalar
        """
        if factors.dim() == 2:
            factors = factors[0]
            target_ret = target_ret[0]

        sig = torch.tanh(factors)
        pos = torch.sign(sig).float()

        prev_pos = torch.roll(pos, 1, dims=0)
        prev_pos[0] = 0.0
        turnover = torch.abs(pos - prev_pos)

        pnl = pos * target_ret - turnover * self.cost_rate

        if pnl.std() < 1e-6:
            return torch.tensor(-2.0, device=pnl.device)

        mean_ret = pnl.mean()
        neg = pnl[pnl < 0]
        downside_std = neg.std() if len(neg) > 5 else pnl.std()
        sortino = mean_ret / (downside_std + 1e-6) * 15.87

        if mean_ret < 0:
            return torch.tensor(-2.0, device=pnl.device)
        if turnover.mean() > 0.5:
            sortino -= 1.0

        return torch.clamp(sortino, -3.0, 5.0)
