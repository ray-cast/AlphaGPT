import torch
import torch.jit


# ---- 基础算子（无依赖，必须最先定义） ----

@torch.jit.script
def _ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    if d == 0: return x
    pad = torch.full((x.shape[0], d), float('nan'), device=x.device)
    return torch.cat([pad, x[:, :-d]], dim=1)


@torch.jit.script
def _signedpower(x: torch.Tensor) -> torch.Tensor:
    """WQ101 核心算子：sign(x) * |x|^0.5，非线性信号放大。"""
    return torch.sign(x) * torch.pow(torch.abs(x) + 1e-8, 0.5)


@torch.jit.script
def _op_gate(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mask = (condition > 0).float()
    return mask * x + (1.0 - mask) * y


@torch.jit.script
def _op_jump(x: torch.Tensor) -> torch.Tensor:
    # expanding window z-score，避免未来数据泄漏
    # 使用 nan-safe 累积和，防止 NaN 永久传播
    N, T = x.shape
    x_safe = torch.where(torch.isnan(x), torch.zeros_like(x), x)
    nan_mask = torch.isnan(x)

    cumsum = torch.cumsum(x_safe, dim=1)
    cumsum2 = torch.cumsum(x_safe * x_safe, dim=1)
    # 每个位置的合法（非NaN）计数
    valid_count = torch.cumsum((~nan_mask).float(), dim=1).clamp(min=1.0)
    arange = valid_count

    mean = cumsum / arange
    var = cumsum2 / arange - mean * mean
    std = torch.sqrt(var.clamp(min=1e-6))
    z = (x_safe - mean) / std
    # 直接输出 z-score，保留连续排序信息
    # 旧版 relu(z - 3.0) 阈值过高，绝大多数输出为 0，
    # 配合 SIGN 后导致信号全部相同，无区分度
    result = z
    # NaN 位置还原为 NaN
    result = torch.where(nan_mask, torch.full_like(result, float('nan')), result)
    return result


@torch.jit.script
def _op_decay(x: torch.Tensor) -> torch.Tensor:
    # 权重归一化，避免深层嵌套时信号爆炸
    w0, w1, w2 = 1.0, 0.8, 0.6
    s = w0 + w1 + w2
    return (w0 * x + w1 * _ts_delay(x, 1) + w2 * _ts_delay(x, 2)) / s


# ---- 时序算子 ----

@torch.jit.script
def _ts_delta(x: torch.Tensor, d: int) -> torch.Tensor:
    """时序变化量：x(t) - x(t-d)。"""
    return x - _ts_delay(x, d)


@torch.jit.script
def _ts_decay_linear(x: torch.Tensor, d: int) -> torch.Tensor:
    """线性衰减均线：近期权重更大。"""
    if d <= 1: return x
    B, T = x.shape
    pad = torch.full((B, d - 1), float('nan'), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    w = torch.arange(1, d + 1, device=x.device, dtype=x.dtype)
    w = w / w.sum()
    return (windows * w).sum(dim=-1)


@torch.jit.script
def _ts_rank(x: torch.Tensor, d: int) -> torch.Tensor:
    """时序排名百分位：当前值在过去 d 天中的排名归一化到 [0, 1]。"""
    if d <= 1: return torch.zeros_like(x)
    B, T = x.shape
    pad = torch.full((B, d - 1), float('nan'), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    current = x.unsqueeze(-1)
    rank = (windows < current).sum(dim=-1).float()
    result = rank / (d - 1)
    # 前 d-1 个窗口含 NaN padding，比较运算不传播 NaN，需显式置 NaN
    result[:, :d - 1] = float('nan')
    return result


@torch.jit.script
def _ts_min(x: torch.Tensor, d: int) -> torch.Tensor:
    """时序滚动最小值。"""
    if d <= 1: return x
    B, T = x.shape
    pad = torch.full((B, d - 1), float('inf'), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    return windows.min(dim=-1)[0]


@torch.jit.script
def _ts_max(x: torch.Tensor, d: int) -> torch.Tensor:
    """时序滚动最大值。"""
    if d <= 1: return x
    B, T = x.shape
    pad = torch.full((B, d - 1), float('-inf'), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    return windows.max(dim=-1)[0]


@torch.jit.script
def _ts_std(x: torch.Tensor, d: int) -> torch.Tensor:
    """时序滚动标准差。使用 cumsum 公式避免 unfold 开销。"""
    if d <= 1: return torch.zeros_like(x)
    B, T = x.shape
    # cumsum 公式: Var(X) = E[X^2] - E[X]^2
    x2 = x * x
    pad = torch.zeros((B, d - 1), device=x.device, dtype=x.dtype)
    cs = torch.cumsum(torch.cat([pad, x], dim=1), dim=1)
    cs2 = torch.cumsum(torch.cat([pad, x2], dim=1), dim=1)
    # 滚动求和
    roll_sum = cs[:, d:] - cs[:, :-d]
    roll_sum2 = cs2[:, d:] - cs2[:, :-d]
    mean = roll_sum / d
    var = roll_sum2 / d - mean * mean
    result = torch.sqrt(var.clamp(min=1e-12))
    # 前 d-1 个位置窗口不足 d，除数错误，显式置 NaN
    result[:, :d - 1] = float('nan')
    return result


@torch.jit.script
def _ts_mean(x: torch.Tensor, d: int) -> torch.Tensor:
    """时序滚动均值。使用 cumsum 公式避免 unfold 开销。"""
    if d <= 1: return x
    B, T = x.shape
    pad = torch.zeros((B, d - 1), device=x.device, dtype=x.dtype)
    cs = torch.cumsum(torch.cat([pad, x], dim=1), dim=1)
    roll_sum = cs[:, d:] - cs[:, :-d]
    result = roll_sum / d
    result[:, :d - 1] = float('nan')
    return result


@torch.jit.script
def _ts_corr(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    """两因子滚动 Pearson 相关系数。使用 cumsum 公式避免 unfold。"""
    if d <= 1: return torch.zeros_like(x)
    B, T = x.shape
    pad = torch.zeros((B, d - 1), device=x.device, dtype=x.dtype)
    xy = x * y
    x2 = x * x
    y2 = y * y
    cs_x = torch.cumsum(torch.cat([pad, x], dim=1), dim=1)
    cs_y = torch.cumsum(torch.cat([pad, y], dim=1), dim=1)
    cs_xy = torch.cumsum(torch.cat([pad, xy], dim=1), dim=1)
    cs_x2 = torch.cumsum(torch.cat([pad, x2], dim=1), dim=1)
    cs_y2 = torch.cumsum(torch.cat([pad, y2], dim=1), dim=1)
    # 滚动求和
    sx = cs_x[:, d:] - cs_x[:, :-d]
    sy = cs_y[:, d:] - cs_y[:, :-d]
    sxy = cs_xy[:, d:] - cs_xy[:, :-d]
    sx2 = cs_x2[:, d:] - cs_x2[:, :-d]
    sy2 = cs_y2[:, d:] - cs_y2[:, :-d]
    # Pearson: cov / (std_x * std_y)
    cov = sxy / d - (sx / d) * (sy / d)
    var_x = (sx2 / d - (sx / d) ** 2).clamp(min=1e-12)
    var_y = (sy2 / d - (sy / d) ** 2).clamp(min=1e-12)
    result = cov / (torch.sqrt(var_x) * torch.sqrt(var_y) + 1e-8)
    result[:, :d - 1] = float('nan')
    return result


@torch.jit.script
def _ts_argmin(x: torch.Tensor, d: int) -> torch.Tensor:
    """时序 argmin：过去 d 天内最小值出现的位置（归一化到 [0, 1]）。"""
    if d <= 1: return torch.zeros_like(x)
    B, T = x.shape
    pad = torch.full((B, d - 1), float('inf'), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    # argmin 位置归一化
    idx = windows.argmin(dim=-1).float() / (d - 1)
    return idx


@torch.jit.script
def _ts_argmax(x: torch.Tensor, d: int) -> torch.Tensor:
    """时序 argmax：过去 d 天内最大值出现的位置（归一化到 [0, 1]）。"""
    if d <= 1: return torch.zeros_like(x)
    B, T = x.shape
    pad = torch.full((B, d - 1), float('-inf'), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    idx = windows.argmax(dim=-1).float() / (d - 1)
    return idx


@torch.jit.script
def _ts_sum(x: torch.Tensor, d: int) -> torch.Tensor:
    """时序滚动求和。使用 cumsum 公式避免 unfold 开销。"""
    if d <= 1: return x
    B, T = x.shape
    pad = torch.zeros((B, d - 1), device=x.device, dtype=x.dtype)
    cs = torch.cumsum(torch.cat([pad, x], dim=1), dim=1)
    result = cs[:, d:] - cs[:, :-d]
    result[:, :d - 1] = float('nan')
    return result


# ---- 截面算子 ----

@torch.jit.script
def _cs_rank(x: torch.Tensor) -> torch.Tensor:
    """截面排名：将每只股票在当日所有股票中的排名归一化到 [0, 1]。"""
    return x.argsort(dim=0).argsort(0).float() / (x.shape[0] + 1e-6)


@torch.jit.script
def _cs_zscore(x: torch.Tensor) -> torch.Tensor:
    """截面 z-score 标准化：保留幅度信息，不同于 rank 只保留排序。"""
    valid = ~torch.isnan(x)
    x_safe = torch.where(valid, x, torch.zeros_like(x))
    n = valid.float().sum(dim=0, keepdim=True).clamp(min=1.0)
    mean = x_safe.sum(dim=0, keepdim=True) / n
    var = (x_safe * x_safe).sum(dim=0, keepdim=True) / n - mean * mean
    std = var.sqrt().clamp(min=1e-6)
    result = (x_safe - mean) / std
    return torch.where(valid, result, torch.zeros_like(result))


# ---- 算子注册表 ----

OPS_CONFIG = [
    # ---- 算术算子 ----
    ('ADD', lambda x, y: x + y, 2),
    ('SUB', lambda x, y: x - y, 2),
    ('MUL', lambda x, y: x * y, 2),
    ('DIV', lambda x, y: x / (y + 1e-6), 2),
    ('NEG', lambda x: -x, 1),
    ('ABS', torch.abs, 1),
    ('SIGN', torch.sign, 1),
    ('SIGNEDPOWER', _signedpower, 1),          # WQ101 核心非线性放大
    # ---- 控制流 ----
    ('GATE', _op_gate, 3),
    # ---- 截面算子 ----
    ('CS_RANK', _cs_rank, 1),
    ('CS_ZSCORE', _cs_zscore, 1),              # 截面z-score保留幅度
    ('CROSS', lambda x, y: _cs_rank(x) * _cs_rank(y), 2),
    # ---- 时序算子（20日窗口） ----
    ('TS_RANK20', lambda x: _ts_rank(x, 20), 1),
    ('TS_DECAY20', lambda x: _ts_decay_linear(x, 20), 1),
    ('TS_STD20', lambda x: _ts_std(x, 20), 1),
    ('TS_CORR20', lambda x, y: _ts_corr(x, y, 20), 2),
    ('TS_MEAN20', lambda x: _ts_mean(x, 20), 1),
    # ---- 时序算子（10日窗口） ----
    ('TS_RANK10', lambda x: _ts_rank(x, 10), 1),
    ('TS_DECAY10', lambda x: _ts_decay_linear(x, 10), 1),
    ('TS_DELTA10', lambda x: _ts_delta(x, 10), 1),
    # ---- 时序算子（5日窗口） ----
    ('TS_DELTA5', lambda x: _ts_delta(x, 5), 1),
    ('TS_MIN5', lambda x: _ts_min(x, 5), 1),
    ('TS_MAX5', lambda x: _ts_max(x, 5), 1),
    ('TS_SUM5', lambda x: _ts_sum(x, 5), 1),  # 滚动求和
    ('TS_ARGMIN5', lambda x: _ts_argmin(x, 5), 1),   # 极值位置
    ('TS_ARGMAX5', lambda x: _ts_argmax(x, 5), 1),
    ('TS_DELAY5', lambda x: _ts_delay(x, 5), 1),
    ('TS_MEAN5', lambda x: _ts_mean(x, 5), 1),
    # ---- 时序算子（60日窗口：中长期趋势） ----
    ('TS_RANK60', lambda x: _ts_rank(x, 60), 1),
    ('TS_DECAY60', lambda x: _ts_decay_linear(x, 60), 1),
    ('TS_STD60', lambda x: _ts_std(x, 60), 1),
    ('TS_CORR60', lambda x, y: _ts_corr(x, y, 60), 2),
    ('TS_MEAN60', lambda x: _ts_mean(x, 60), 1),
    ('TS_DELTA60', lambda x: _ts_delta(x, 60), 1),
]
