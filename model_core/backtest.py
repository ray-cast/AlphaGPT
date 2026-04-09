import torch
from .config import ModelConfig


# ---- JIT 编译的纯函数 ----

@torch.jit.script
def _vectorized_ic_raw(factors: torch.Tensor, target_ret: torch.Tensor,
                       valid_mask: torch.Tensor) -> torch.Tensor:
    """向量化计算逐日 Pearson IC，返回原始均值。"""
    n_valid = valid_mask.float().sum(dim=0).clamp(min=1)

    f_c = torch.where(valid_mask, factors, 0.0)
    r_c = torch.where(valid_mask, target_ret, 0.0)
    f_mean = f_c.sum(dim=0) / n_valid
    r_mean = r_c.sum(dim=0) / n_valid
    f_c = torch.where(valid_mask, f_c - f_mean.unsqueeze(0), 0.0)
    r_c = torch.where(valid_mask, r_c - r_mean.unsqueeze(0), 0.0)

    cov = (f_c * r_c).sum(dim=0)
    std_f = (f_c.pow(2).sum(dim=0)).sqrt().clamp(min=1e-6)
    std_r = (r_c.pow(2).sum(dim=0)).sqrt().clamp(min=1e-6)
    daily_ic = cov / (std_f * std_r)

    valid_day = (n_valid >= 10) & (std_f > 1e-3) & (std_r > 1e-3)
    ic_count = valid_day.float().sum()
    if ic_count < 1:
        return torch.tensor(0.0, device=factors.device)

    avg_ic = (daily_ic * valid_day.float()).sum() / ic_count
    return avg_ic


class AshareBacktest:
    """A股截面选股回测引擎。

    每个交易日对所有股票按 alpha 分值排序，做多排名前 N 只。
    已内置 T+1 规则（通过 open-to-open target_ret 实现）、佣金、换手率过滤。
    """

    def __init__(self):
        self.commission = ModelConfig.COMMISSION_RATE
        self.min_turnover = ModelConfig.MIN_TURNOVER_RATE
        self.top_n = ModelConfig.TOP_N_STOCKS

    def evaluate(self, factors, raw_data, target_ret, start_idx=0, end_idx=None):
        """
        Args:
            factors:    [num_stocks, T] alpha 信号（来自 PrefixVM）
            raw_data:   dict，含 'turnover_rate', 'suspended', 'ipo_ok'
            target_ret: [num_stocks, T] open-to-open 前向收益
            start_idx:  起始时间索引（含）
            end_idx:    结束时间索引（不含），None 表示到末尾
        Returns:
            (fitness: scalar tensor, cum_ret: float, daily_pnl: tensor)
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
        scores = torch.where(tradeable, factors, torch.tensor(float('-inf'), device=factors.device))

        # 持仓：每日 top-K
        position = torch.zeros_like(scores)
        _, topk_idx = scores.topk(self.top_n, dim=0)
        position.scatter_(0, topk_idx, 1.0)

        # PnL（含佣金）
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0.0
        tx_cost = torch.abs(position - prev_pos) * self.commission
        daily_pnl = (position * target_ret - tx_cost).sum(dim=0) / self.top_n

        # 评分：IC × 10
        if N_stocks > 10:
            fitness = _vectorized_ic_raw(factors, target_ret, valid_mask) * 10.0
        else:
            fitness = torch.tensor(0.0, device=factors.device)

        return fitness, daily_pnl.sum().item(), daily_pnl
