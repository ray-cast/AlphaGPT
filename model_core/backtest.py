import torch
from .config import ModelConfig


class AshareBacktest:
    """A股截面选股回测引擎。

    每个交易日对所有股票按 alpha 分值排序，做多排名前 N 只。
    已内置 T+1 规则（通过 open-to-open target_ret 实现）、佣金、换手率过滤。
    """

    def __init__(self):
        self.commission = ModelConfig.COMMISSION_RATE
        self.min_turnover = ModelConfig.MIN_TURNOVER_RATE
        self.top_n = ModelConfig.TOP_N_STOCKS

    def evaluate(self, factors, raw_data, target_ret, start_idx=0, end_idx=None, train_step=0):
        """
        Args:
            factors:    [num_stocks, T] alpha 信号（来自 PrefixVM）
            raw_data:   dict，含 'turnover_rate', 'suspended', 'ipo_ok'
            target_ret: [num_stocks, T] open-to-open 前向收益
            start_idx:  起始时间索引（含）
            end_idx:    结束时间索引（不含），None 表示到末尾
        Returns:
            (fitness: scalar tensor, cum_ret: float, daily_pnl: tensor, sharpe: float)
        """
        turnover_rate = raw_data.get("turnover_rate",
                            torch.zeros_like(factors))
        suspended = raw_data.get("suspended",
                            torch.zeros_like(factors, dtype=torch.bool))
        constituent = raw_data.get("ipo_ok",
                            torch.ones_like(factors, dtype=torch.bool))

        # 按指定区间切片
        factors = factors[:, start_idx:end_idx]
        target_ret = target_ret[:, start_idx:end_idx]
        turnover_rate = turnover_rate[:, start_idx:end_idx]
        suspended = suspended[:, start_idx:end_idx]
        constituent = constituent[:, start_idx:end_idx]

        N_stocks, T_len = factors.shape

        # 排除停牌/非成分股
        valid_mask = ~(torch.isnan(factors) | torch.isnan(target_ret) | suspended) & constituent
        tradeable = valid_mask & (turnover_rate > self.min_turnover)
        tradeable_f = tradeable.float()
        scores = torch.where(tradeable, factors, torch.tensor(float('-inf'), device=factors.device))

        # 持仓：每日 top-K
        position = torch.zeros_like(scores)
        _, topk_idx = torch.topk(scores, self.top_n, dim=0)
        position.scatter_(0, topk_idx, 1.0)

        # PnL（含佣金）
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0.0
        tx_cost = torch.abs(position - prev_pos) * self.commission
        daily_pnl = (position * target_ret - tx_cost).sum(dim=0) / (self.top_n + 1e-9)

        # ---- Fitness 计算 ----
        # 1) Sortino 比率作为主指标
        mean_ret = daily_pnl.mean()
        neg_returns = torch.where(daily_pnl < 0, daily_pnl, 0.0)
        if (daily_pnl < 0).any():
            downside_std = neg_returns.std()
        else:
            downside_std = daily_pnl.std() + 1e-6

        sortino = mean_ret / (downside_std + 1e-6)

        # 计算夏普比率
        cum_ret = daily_pnl.sum().item()
        if T_len > 0:
            ann_ret = (1 + cum_ret) ** (252 / T_len) - 1
            ann_vol = daily_pnl.std().item() * (252 ** 0.5)
            sharpe = (ann_ret - 0.02) / (ann_vol + 1e-6)
        else:
            sharpe = 0.0

        # ---- IC 计算（截面 Spearman rank correlation）----
        f_scored = factors.masked_fill(~tradeable, float('-inf'))
        r_scored = target_ret.masked_fill(~tradeable, float('-inf'))
        # scatter 单次排序替代 double argsort
        arange_row = torch.arange(N_stocks, device=factors.device).float().unsqueeze(1)
        f_idx = f_scored.argsort(dim=0)
        r_idx = r_scored.argsort(dim=0)
        f_rank = torch.zeros_like(f_scored).scatter_(0, f_idx, arange_row.expand_as(f_idx))
        r_rank = torch.zeros_like(r_scored).scatter_(0, r_idx, arange_row.expand_as(r_idx))
        valid_count = tradeable_f.sum(dim=0)
        f_mean = (f_rank * tradeable_f).sum(dim=0) / valid_count.clamp(min=1)
        r_mean = (r_rank * tradeable_f).sum(dim=0) / valid_count.clamp(min=1)
        f_c = (f_rank - f_mean.unsqueeze(0)) * tradeable_f
        r_c = (r_rank - r_mean.unsqueeze(0)) * tradeable_f
        cov = (f_c * r_c).sum(dim=0)
        ic_per_day = cov / (f_c.norm(dim=0) * r_c.norm(dim=0) + 1e-8)
        std_f = (f_c.pow(2).sum(dim=0)).sqrt()
        std_r = (r_c.pow(2).sum(dim=0)).sqrt()
        valid_days = (valid_count >= 20) & (std_f > 1e-3) & (std_r > 1e-3)
        mean_ic = ic_per_day[valid_days].mean().item() if valid_days.any() else 0.0
        ic_std = ic_per_day[valid_days].std().item() if valid_days.sum() > 1 else 1.0
        ir = mean_ic / (ic_std + 1e-8)

        # 综合得分（QFR: IC̄ − λ · 𝟙{IR ≤ clip[(step − α)·η, 0, δ]}）
        clip_val = min(max((train_step - ModelConfig.QFR_ALPHA) * ModelConfig.QFR_ETA, 0.0), ModelConfig.QFR_DELTA)
        ir_penalty = ModelConfig.QFR_LAMBDA if ir <= clip_val else 0.0
        fitness = sortino.item() + ModelConfig.IC_WEIGHT * (mean_ic - ir_penalty)

        return fitness, cum_ret, daily_pnl, sharpe, mean_ic, ir
