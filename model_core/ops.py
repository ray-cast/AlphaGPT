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
    """时序滚动标准差。使用 unfold 避免 cumsum NaN 永久传播。"""
    if d <= 1: return torch.zeros_like(x)
    B, T = x.shape
    pad = torch.full((B, d - 1), float('nan'), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    mean = windows.mean(dim=-1, keepdim=True)
    var = ((windows - mean) ** 2).sum(dim=-1) / d
    return torch.sqrt(var.clamp(min=1e-12))


@torch.jit.script
def _ts_mean(x: torch.Tensor, d: int) -> torch.Tensor:
    """时序滚动均值。"""
    if d <= 1: return x
    B, T = x.shape
    pad = torch.full((B, d - 1), float('nan'), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    return windows.mean(dim=-1)


@torch.jit.script
def _ts_corr(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    """两因子滚动 Pearson 相关系数。使用 unfold 避免 cumsum NaN 传播。"""
    if d <= 1: return torch.zeros_like(x)
    B, T = x.shape
    pad = torch.full((B, d - 1), float('nan'), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    y_pad = torch.cat([pad, y], dim=1)
    wx = x_pad.unfold(1, d, 1)
    wy = y_pad.unfold(1, d, 1)
    mx = wx.mean(dim=-1, keepdim=True)
    my = wy.mean(dim=-1, keepdim=True)
    cov = ((wx - mx) * (wy - my)).sum(dim=-1) / d
    var_x = ((wx - mx) ** 2).sum(dim=-1) / d
    var_y = ((wy - my) ** 2).sum(dim=-1) / d
    result = cov / (torch.sqrt(var_x.clamp(min=1e-12)) * torch.sqrt(var_y.clamp(min=1e-12)) + 1e-8)
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
    """时序滚动求和。"""
    if d <= 1: return x
    B, T = x.shape
    pad = torch.full((B, d - 1), float('nan'), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    return windows.sum(dim=-1)


@torch.jit.script
def _ts_ref(x: torch.Tensor, d: int) -> torch.Tensor:
    """引用 d 期前的值。"""
    return _ts_delay(x, d)


@torch.jit.script
def _ts_var(x: torch.Tensor, d: int) -> torch.Tensor:
    """时序滚动方差。"""
    if d <= 1:
        return torch.zeros_like(x)
    B, T = x.shape
    pad = torch.full((B, d - 1), float('nan'), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    mean = windows.mean(dim=-1, keepdim=True)
    return ((windows - mean) ** 2).sum(dim=-1) / d


@torch.jit.script
def _ts_skew(x: torch.Tensor, d: int) -> torch.Tensor:
    """时序滚动偏度。"""
    if d <= 2:
        return torch.zeros_like(x)
    B, T = x.shape
    pad = torch.full((B, d - 1), float('nan'), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    mean = windows.mean(dim=-1, keepdim=True)
    diff = windows - mean
    m2 = (diff ** 2).sum(dim=-1) / d
    m3 = (diff ** 3).sum(dim=-1) / d
    return m3 / torch.pow(m2.clamp(min=1e-12), 1.5)


@torch.jit.script
def _ts_kurt(x: torch.Tensor, d: int) -> torch.Tensor:
    """时序滚动超额峰度。"""
    if d <= 3:
        return torch.zeros_like(x)
    B, T = x.shape
    pad = torch.full((B, d - 1), float('nan'), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    mean = windows.mean(dim=-1, keepdim=True)
    diff = windows - mean
    m2 = (diff ** 2).sum(dim=-1) / d
    m4 = (diff ** 4).sum(dim=-1) / d
    return m4 / m2.clamp(min=1e-12).pow(2) - 3.0


@torch.jit.script
def _ts_med(x: torch.Tensor, d: int) -> torch.Tensor:
    """时序滚动中位数。"""
    if d <= 1:
        return x
    B, T = x.shape
    pad = torch.full((B, d - 1), float('nan'), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    return windows.sort(dim=-1)[0][:, :, d // 2]


@torch.jit.script
def _ts_mad(x: torch.Tensor, d: int) -> torch.Tensor:
    """时序滚动平均绝对偏差 MAD = mean(|x - mean(x)|)。"""
    if d <= 1:
        return torch.zeros_like(x)
    B, T = x.shape
    pad = torch.full((B, d - 1), float('nan'), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    mean = windows.mean(dim=-1, keepdim=True)
    return (windows - mean).abs().mean(dim=-1)


@torch.jit.script
def _ts_wma(x: torch.Tensor, d: int) -> torch.Tensor:
    """加权移动平均：线性递增权重 [1, 2, ..., d]。"""
    if d <= 1:
        return x
    B, T = x.shape
    pad = torch.full((B, d - 1), float('nan'), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    w = torch.arange(1, d + 1, device=x.device, dtype=x.dtype)
    w = w / w.sum()
    return (windows * w).sum(dim=-1)


@torch.jit.script
def _ts_ema(x: torch.Tensor, d: int) -> torch.Tensor:
    """指数移动平均：alpha=2/(d+1)，几何衰减权重近似。"""
    if d <= 1:
        return x
    B, T = x.shape
    alpha = 2.0 / (d + 1.0)
    pad = torch.full((B, d - 1), float('nan'), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    powers = torch.arange(d - 1, -1, -1, device=x.device, dtype=x.dtype)
    w = alpha * torch.pow(1.0 - alpha, powers)
    w = w / w.sum()
    return (windows * w).sum(dim=-1)


@torch.jit.script
def _ts_cov(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    """两因子滚动协方差。"""
    if d <= 1:
        return torch.zeros_like(x)
    B, T = x.shape
    pad = torch.full((B, d - 1), float('nan'), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    y_pad = torch.cat([pad, y], dim=1)
    wx = x_pad.unfold(1, d, 1)
    wy = y_pad.unfold(1, d, 1)
    mx = wx.mean(dim=-1, keepdim=True)
    my = wy.mean(dim=-1, keepdim=True)
    return ((wx - mx) * (wy - my)).sum(dim=-1) / d


# ---- 算子注册表 ----

OPS_CONFIG = [
    # ---- 一元算子 ----
    ('ABS', torch.abs, 1),
    ('SIGN', torch.sign, 1),
    ('LOG', lambda x: torch.log(x.abs() + 1e-8), 1),
    ('NEG', lambda x: -x, 1),
    ('SIGNEDPOWER', _signedpower, 1),
    # ---- 二元算子 ----
    ('ADD', lambda x, y: x + y, 2),
    ('SUB', lambda x, y: x - y, 2),
    ('MUL', lambda x, y: x * y, 2),
    ('DIV', lambda x, y: x / (y + 1e-6), 2),
    ('POW', lambda x, y: torch.pow(x.abs() + 1e-8, y), 2),
    ('GREATER', lambda x, y: (x > y).float(), 2),
    ('LESS', lambda x, y: (x < y).float(), 2),
    # ---- 常量（arity=0，返回标量，运算时自动广播） ----
    ('CONST_-30', lambda: -30.0, 0),
    ('CONST_-20', lambda: -20.0, 0),
    ('CONST_-10', lambda: -10.0, 0),
    ('CONST_-5', lambda: -5.0, 0),
    ('CONST_-2', lambda: -2.0, 0),
    ('CONST_-1', lambda: -1.0, 0),
    ('CONST_-0.5', lambda: -0.5, 0),
    ('CONST_0.5', lambda: 0.5, 0),
    ('CONST_1', lambda: 1.0, 0),
    ('CONST_2', lambda: 2.0, 0),
    ('CONST_5', lambda: 5.0, 0),
    ('CONST_10', lambda: 10.0, 0),
    ('CONST_20', lambda: 20.0, 0),
    ('CONST_30', lambda: 30.0, 0),
    # ---- 时序算子（5日窗口） ----
    ('TS_REF5', lambda x: _ts_ref(x, 5), 1),
    ('TS_MEAN5', lambda x: _ts_mean(x, 5), 1),
    ('TS_SUM5', lambda x: _ts_sum(x, 5), 1),
    ('TS_STD5', lambda x: _ts_std(x, 5), 1),
    ('TS_VAR5', lambda x: _ts_var(x, 5), 1),
    ('TS_SKEW5', lambda x: _ts_skew(x, 5), 1),
    ('TS_KURT5', lambda x: _ts_kurt(x, 5), 1),
    ('TS_MAX5', lambda x: _ts_max(x, 5), 1),
    ('TS_MIN5', lambda x: _ts_min(x, 5), 1),
    ('TS_MED5', lambda x: _ts_med(x, 5), 1),
    ('TS_MAD5', lambda x: _ts_mad(x, 5), 1),
    ('TS_RANK5', lambda x: _ts_rank(x, 5), 1),
    ('TS_DELTA5', lambda x: _ts_delta(x, 5), 1),
    ('TS_WMA5', lambda x: _ts_wma(x, 5), 1),
    ('TS_EMA5', lambda x: _ts_ema(x, 5), 1),
    ('TS_DELAY5', lambda x: _ts_delay(x, 5), 1),
    ('TS_ARGMIN5', lambda x: _ts_argmin(x, 5), 1),
    ('TS_ARGMAX5', lambda x: _ts_argmax(x, 5), 1),
    ('TS_COV5', lambda x, y: _ts_cov(x, y, 5), 2),
    ('TS_CORR5', lambda x, y: _ts_corr(x, y, 5), 2),
    # ---- 时序算子（10日窗口） ----
    ('TS_REF10', lambda x: _ts_ref(x, 10), 1),
    ('TS_MEAN10', lambda x: _ts_mean(x, 10), 1),
    ('TS_SUM10', lambda x: _ts_sum(x, 10), 1),
    ('TS_STD10', lambda x: _ts_std(x, 10), 1),
    ('TS_VAR10', lambda x: _ts_var(x, 10), 1),
    ('TS_SKEW10', lambda x: _ts_skew(x, 10), 1),
    ('TS_KURT10', lambda x: _ts_kurt(x, 10), 1),
    ('TS_MAX10', lambda x: _ts_max(x, 10), 1),
    ('TS_MIN10', lambda x: _ts_min(x, 10), 1),
    ('TS_MED10', lambda x: _ts_med(x, 10), 1),
    ('TS_MAD10', lambda x: _ts_mad(x, 10), 1),
    ('TS_RANK10', lambda x: _ts_rank(x, 10), 1),
    ('TS_DELTA10', lambda x: _ts_delta(x, 10), 1),
    ('TS_DECAY10', lambda x: _ts_decay_linear(x, 10), 1),
    ('TS_WMA10', lambda x: _ts_wma(x, 10), 1),
    ('TS_EMA10', lambda x: _ts_ema(x, 10), 1),
    ('TS_COV10', lambda x, y: _ts_cov(x, y, 10), 2),
    ('TS_CORR10', lambda x, y: _ts_corr(x, y, 10), 2),
    # ---- 时序算子（20日窗口） ----
    ('TS_REF20', lambda x: _ts_ref(x, 20), 1),
    ('TS_MEAN20', lambda x: _ts_mean(x, 20), 1),
    ('TS_SUM20', lambda x: _ts_sum(x, 20), 1),
    ('TS_STD20', lambda x: _ts_std(x, 20), 1),
    ('TS_VAR20', lambda x: _ts_var(x, 20), 1),
    ('TS_SKEW20', lambda x: _ts_skew(x, 20), 1),
    ('TS_KURT20', lambda x: _ts_kurt(x, 20), 1),
    ('TS_MAX20', lambda x: _ts_max(x, 20), 1),
    ('TS_MIN20', lambda x: _ts_min(x, 20), 1),
    ('TS_MED20', lambda x: _ts_med(x, 20), 1),
    ('TS_MAD20', lambda x: _ts_mad(x, 20), 1),
    ('TS_RANK20', lambda x: _ts_rank(x, 20), 1),
    ('TS_DECAY20', lambda x: _ts_decay_linear(x, 20), 1),
    ('TS_WMA20', lambda x: _ts_wma(x, 20), 1),
    ('TS_EMA20', lambda x: _ts_ema(x, 20), 1),
    ('TS_COV20', lambda x, y: _ts_cov(x, y, 20), 2),
    ('TS_CORR20', lambda x, y: _ts_corr(x, y, 20), 2),
    # ---- 时序算子（40日窗口：中长期趋势） ----
    ('TS_REF40', lambda x: _ts_ref(x, 40), 1),
    ('TS_MEAN40', lambda x: _ts_mean(x, 40), 1),
    ('TS_SUM40', lambda x: _ts_sum(x, 40), 1),
    ('TS_STD40', lambda x: _ts_std(x, 40), 1),
    ('TS_VAR40', lambda x: _ts_var(x, 40), 1),
    ('TS_SKEW40', lambda x: _ts_skew(x, 40), 1),
    ('TS_KURT40', lambda x: _ts_kurt(x, 40), 1),
    ('TS_MAX40', lambda x: _ts_max(x, 40), 1),
    ('TS_MIN40', lambda x: _ts_min(x, 40), 1),
    ('TS_MED40', lambda x: _ts_med(x, 40), 1),
    ('TS_MAD40', lambda x: _ts_mad(x, 40), 1),
    ('TS_RANK40', lambda x: _ts_rank(x, 40), 1),
    ('TS_DECAY40', lambda x: _ts_decay_linear(x, 40), 1),
    ('TS_DELTA40', lambda x: _ts_delta(x, 40), 1),
    ('TS_WMA40', lambda x: _ts_wma(x, 40), 1),
    ('TS_EMA40', lambda x: _ts_ema(x, 40), 1),
    ('TS_COV40', lambda x, y: _ts_cov(x, y, 40), 2),
    ('TS_CORR40', lambda x, y: _ts_corr(x, y, 40), 2),
]
