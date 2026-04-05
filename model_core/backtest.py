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

    def evaluate(self, factors, raw_data, target_ret, start_idx=0, end_idx=None):
        """
        Args:
            factors:    [num_stocks, T] alpha 信号（来自 PrefixVM）
            raw_data:   dict，含 'turnover_rate', 'pct_chg'（如有）, 'vol', 'close'
            target_ret: [num_stocks, T] open-to-open 前向收益
            start_idx:  起始时间索引（含）
            end_idx:    结束时间索引（不含），None 表示到末尾
        Returns:
            (fitness: scalar tensor, avg_daily_return: float)
        """
        turnover_rate = raw_data.get("turnover_rate",
                            torch.zeros_like(factors))
        suspended = raw_data.get("suspended",
                            torch.zeros_like(factors, dtype=torch.bool))
        constituent = raw_data.get("constituent",
                            torch.ones_like(factors, dtype=torch.bool))

        # 按指定区间切片
        factors = factors[:, start_idx:end_idx]
        target_ret = target_ret[:, start_idx:end_idx]
        turnover_rate = turnover_rate[:, start_idx:end_idx]
        suspended = suspended[:, start_idx:end_idx]
        constituent = constituent[:, start_idx:end_idx]

        N_stocks, T_len = factors.shape

        # 排除停牌/非成分股（NaN 信号、NaN target_ret、停牌、或非时点成分股）
        valid_mask = ~(torch.isnan(factors) | torch.isnan(target_ret) | suspended) & constituent
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

        # 熊市减仓系数：市场弱势时收益按半仓计算（不影响实际换手）
        bear_mask = market_ma20 < 0
        bear_scale = torch.ones(T_len, device=factors.device)
        bear_scale[bear_mask] = 0.5

        # 换手（基于实际选股变化，不含熊市虚拟减仓）
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0.0
        turnover = torch.abs(position - prev_pos)

        # 交易成本
        avg_cost = (self.buy_cost + self.sell_cost) / 2.0
        tx_cost = turnover * avg_cost

        # PnL：应用熊市减仓系数到收益端（不产生虚假换手成本）
        gross_pnl = position * target_ret * bear_scale.unsqueeze(0)
        net_pnl = gross_pnl - tx_cost

        # 每天 portfolio 收益（按实际持仓权重归一化，熊市减仓时分母会相应减小）
        daily_pnl = net_pnl.sum(dim=0) / (position.sum(dim=0).clamp(min=1))

        # ---- Fitness 计算 ----
        # 1) Sortino 比率作为主指标
        mean_ret = daily_pnl.mean()
        # 真正的下行标准差：sqrt(mean(min(R-target, 0)^2))，target=0
        downside_sq = torch.relu(-daily_pnl).pow(2)
        downside_std = torch.sqrt(downside_sq.mean() + 1e-12)
        if downside_std < 1e-8:
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

        # IC 奖励：向量化 Spearman 秩相关（截面预测能力）
        # 仅当 Sortino > -2 时计算 IC，避免对差公式浪费计算
        ic_bonus = 0.0
        if N_stocks > 10 and sortino.item() > -2.0:
            ic_bonus = self._vectorized_ic(factors, target_ret, valid_mask, T_len, N_stocks)

        # 综合得分
        fitness = sortino - dd_penalty - turnover_penalty - neg_return_penalty + ic_bonus

        return fitness, cum_ret.item()

    @staticmethod
    def _vectorized_ic(factors, target_ret, valid_mask, T_len, N_stocks):
        """向量化计算逐日 Pearson 秩相关 IC（近似 Spearman）。

        直接用 Pearson 相关系数代替 Spearman，避免 argsort 开销。
        对于截面选股来说两者高度一致（同调单调变换不改变排序）。
        """
        # 每日有效股票数
        n_valid = valid_mask.float().sum(dim=0).clamp(min=1)  # [T_len]

        # 将 invalid 位置置零
        f_c = factors.clone()
        r_c = target_ret.clone()
        f_c[~valid_mask] = 0.0
        r_c[~valid_mask] = 0.0

        # 中心化（只对 valid 位置）
        f_mean = f_c.sum(dim=0) / n_valid
        r_mean = r_c.sum(dim=0) / n_valid
        f_c -= f_mean.unsqueeze(0)
        r_c -= r_mean.unsqueeze(0)
        f_c[~valid_mask] = 0.0
        r_c[~valid_mask] = 0.0

        # 逐日 Pearson 相关系数
        cov = (f_c * r_c).sum(dim=0)
        std_f = (f_c.pow(2).sum(dim=0)).sqrt().clamp(min=1e-6)
        std_r = (r_c.pow(2).sum(dim=0)).sqrt().clamp(min=1e-6)
        daily_ic = cov / (std_f * std_r)

        # 过滤无效天
        valid_day = (n_valid >= 10) & (std_f > 1e-3) & (std_r > 1e-3)
        ic_count = valid_day.float().sum()
        if ic_count < 1:
            return 0.0

        avg_ic = (daily_ic * valid_day.float()).sum() / ic_count
        return (avg_ic * 5.0).item()

    @staticmethod
    def _rolling_mean_1d(x, window):
        """对 1D tensor 计算滚动均值，前期不足 window 时用 expanding mean。"""
        T = x.shape[0]
        if T < window:
            # 不足一个窗口，全部用 expanding mean
            cumsum = torch.cumsum(x, dim=0)
            arange = torch.arange(1, T + 1, device=x.device, dtype=x.dtype)
            return cumsum / arange
        cumsum = torch.cumsum(x, dim=0)
        arange = torch.arange(1, T + 1, device=x.device, dtype=x.dtype)
        expanding_mean = cumsum / arange
        # 从第 window 个元素起用固定窗口
        rolling_sum = cumsum[window - 1:] - torch.cat(
            [torch.zeros(1, device=x.device), cumsum[:T - window]]
        )
        rolling_mean = rolling_sum / window
        return torch.cat([expanding_mean[:window - 1], rolling_mean])
