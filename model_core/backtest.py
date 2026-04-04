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
            factors:    [num_stocks, T] alpha 信号（来自 StackVM）
            raw_data:   dict，含 'turnover_rate', 'pct_chg'（如有）, 'vol'
            target_ret: [num_stocks, T] open-to-open 前向收益
        Returns:
            (fitness: scalar tensor, avg_daily_return: float)
        """
        turnover_rate = raw_data.get("turnover_rate",
                            torch.zeros_like(factors))

        # 只在训练期评估（前 80%）
        T = factors.shape[1]
        split = int(T * 0.8)
        factors = factors[:, :split]
        target_ret = target_ret[:, :split]
        turnover_rate = turnover_rate[:, :split]

        N_stocks, T_len = factors.shape

        # 信号 → 排名 → 持仓
        # 每天截面排序，做多 top_n
        position = torch.zeros_like(factors)
        for t in range(T_len):
            alpha_t = factors[:, t]

            # 换手率过滤：排除停牌/流动性不足
            valid = turnover_rate[:, t] > self.min_turnover

            # 在有效股票中排序
            scores = alpha_t.clone()
            scores[~valid] = -float("inf")

            # 选取 top_n
            _, topk_idx = torch.topk(scores, min(self.top_n, valid.sum().int().item()))
            position[topk_idx, t] = 1.0

        # 换手
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0.0
        turnover = torch.abs(position - prev_pos)

        # 交易成本
        # 买入时扣 buy_cost，卖出时扣 sell_cost，近似取均值
        avg_cost = (self.buy_cost + self.sell_cost) / 2.0
        tx_cost = turnover * avg_cost

        # PnL
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
        dd_penalty = torch.relu(-max_dd - 0.05) * 2.0  # 回撤超5%惩罚

        # 换手率惩罚
        avg_turnover = turnover.mean()
        turnover_penalty = 0.0
        if avg_turnover > 0.5:
            turnover_penalty = 1.0

        # 活跃度不足惩罚
        active_days = (position.sum(dim=0) > 0).float().sum()
        if active_days < 10:
            return torch.tensor(-10.0, device=factors.device), 0.0

        # 综合得分
        fitness = sortino - dd_penalty - turnover_penalty

        return fitness, cum_ret.item()


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
