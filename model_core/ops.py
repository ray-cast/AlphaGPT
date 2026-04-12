import torch
import torch.jit


# ---- 基础算子（无依赖，必须最先定义） ----

@torch.jit.script
def _ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    if d == 0: return x
    pad = torch.full((x.shape[0], d), float('nan'), device=x.device)
    return torch.cat([pad, x[:, :-d]], dim=1)


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


# ---- 窗口提取 ----

VALID_WINDOWS = [5, 10, 20, 40]


def _snap_window(w):
    """从 tensor 参数提取窗口大小，snap 到最近合法值。"""
    val = abs(float(w[0, 0]))
    if val != val:  # NaN
        return 20
    return min(VALID_WINDOWS, key=lambda d: abs(d - val))


# ---- 算子注册表 ----
# 时序算子窗口由参数决定（第二/三参数），不再硬编码窗口大小。
# 词汇表从 99 缩减到 ~47，模型搜索更高效。

OPS_CONFIG = [
    # ---- 一元算子 ----
    ('ABS', torch.abs, 1),
    ('SIGN', torch.sign, 1),
    ('LOG', lambda x: torch.log(x.abs() + 1e-8), 1),
    ('NEG', lambda x: -x, 1),
        # ---- 二元算子 ----
    ('ADD', lambda x, y: x + y, 2),
    ('SUB', lambda x, y: x - y, 2),
    ('MUL', lambda x, y: x * y, 2),
    ('DIV', lambda x, y: x / (y + 1e-6), 2),
    ('POW', lambda x, y: torch.pow(x.abs() + 1e-8, y), 2),
    ('GREATER', lambda x, y: (x > y).float(), 2),
    ('LESS', lambda x, y: (x < y).float(), 2),
    ('MIN', lambda x, y: torch.min(x, y), 2),
    ('MAX', lambda x, y: torch.max(x, y), 2),
    # ---- 常量（arity=0，返回标量，运算时自动广播） ----
    ('CONST_-30', lambda: -30.0, 0),
    ('CONST_-20', lambda: -20.0, 0),
    ('CONST_-10', lambda: -10.0, 0),
    ('CONST_-5', lambda: -5.0, 0),
    ('CONST_-2', lambda: -2.0, 0),
    ('CONST_-1', lambda: -1.0, 0),
    ('CONST_-0.5', lambda: -0.5, 0),
    ('CONST_0', lambda: 0.0, 0),
    ('CONST_0.5', lambda: 0.5, 0),
    ('CONST_1', lambda: 1.0, 0),
    ('CONST_2', lambda: 2.0, 0),
    ('CONST_5', lambda: 5.0, 0),
    ('CONST_10', lambda: 10.0, 0),
    ('CONST_20', lambda: 20.0, 0),
    ('CONST_40', lambda: 40.0, 0),
    # ---- 时序算子（单序列，arity=2，第二参数决定窗口） ----
    ('TS_REF', lambda x, w: _ts_ref(x, _snap_window(w)), 2),
    ('TS_MEAN', lambda x, w: _ts_mean(x, _snap_window(w)), 2),
    ('TS_SUM', lambda x, w: _ts_sum(x, _snap_window(w)), 2),
    ('TS_STD', lambda x, w: _ts_std(x, _snap_window(w)), 2),
    ('TS_VAR', lambda x, w: _ts_var(x, _snap_window(w)), 2),
    ('TS_SKEW', lambda x, w: _ts_skew(x, _snap_window(w)), 2),
    ('TS_KURT', lambda x, w: _ts_kurt(x, _snap_window(w)), 2),
    ('TS_MAX', lambda x, w: _ts_max(x, _snap_window(w)), 2),
    ('TS_MIN', lambda x, w: _ts_min(x, _snap_window(w)), 2),
    ('TS_MED', lambda x, w: _ts_med(x, _snap_window(w)), 2),
    ('TS_MAD', lambda x, w: _ts_mad(x, _snap_window(w)), 2),
    ('TS_RANK', lambda x, w: _ts_rank(x, _snap_window(w)), 2),
    ('TS_DELTA', lambda x, w: _ts_delta(x, _snap_window(w)), 2),
    ('TS_DECAY', lambda x, w: _ts_decay_linear(x, _snap_window(w)), 2),
    ('TS_WMA', lambda x, w: _ts_wma(x, _snap_window(w)), 2),
    ('TS_EMA', lambda x, w: _ts_ema(x, _snap_window(w)), 2),
    ('TS_ARGMIN', lambda x, w: _ts_argmin(x, _snap_window(w)), 2),
    ('TS_ARGMAX', lambda x, w: _ts_argmax(x, _snap_window(w)), 2),
    # ---- 双序列时序算子（arity=3，第三参数决定窗口） ----
    ('TS_COV', lambda x, y, w: _ts_cov(x, y, _snap_window(w)), 3),
    ('TS_CORR', lambda x, y, w: _ts_corr(x, y, _snap_window(w)), 3),
]
