import torch
import torch.jit


def _ts_delta(x: torch.Tensor, d: int) -> torch.Tensor:
    """时序变化量：x(t) - x(t-d)。"""
    return x - _ts_delay(x, d)


def _ts_decay_linear(x: torch.Tensor, d: int) -> torch.Tensor:
    """线性衰减均线：近期权重更大。"""
    if d <= 1: return x
    B, T = x.shape
    pad = torch.zeros((B, d - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    w = torch.arange(1, d + 1, device=x.device, dtype=x.dtype)
    w = w / w.sum()
    return (windows * w).sum(dim=-1)


def _ts_rank(x: torch.Tensor, d: int) -> torch.Tensor:
    """时序排名百分位：当前值在过去 d 天中的排名归一化到 [0, 1]。"""
    if d <= 1: return torch.zeros_like(x)
    B, T = x.shape
    pad = torch.zeros((B, d - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    current = x.unsqueeze(-1)
    rank = (windows < current).sum(dim=-1).float()
    return rank / (d - 1)


def _ts_min(x: torch.Tensor, d: int) -> torch.Tensor:
    """时序滚动最小值。"""
    if d <= 1: return x
    B, T = x.shape
    pad = torch.full((B, d - 1), float('inf'), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    return windows.min(dim=-1)[0]


def _ts_max(x: torch.Tensor, d: int) -> torch.Tensor:
    """时序滚动最大值。"""
    if d <= 1: return x
    B, T = x.shape
    pad = torch.full((B, d - 1), float('-inf'), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    return windows.max(dim=-1)[0]


def _ts_std(x: torch.Tensor, d: int) -> torch.Tensor:
    """时序滚动标准差。"""
    if d <= 1: return torch.zeros_like(x)
    B, T = x.shape
    pad = torch.zeros((B, d - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    return windows.std(dim=-1)


def _ts_corr(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    """两因子滚动 Pearson 相关系数。"""
    if d <= 1: return torch.zeros_like(x)
    B, T = x.shape
    pad = torch.zeros((B, d - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    y_pad = torch.cat([pad, y], dim=1)
    xw = x_pad.unfold(1, d, 1)
    yw = y_pad.unfold(1, d, 1)
    xm = xw - xw.mean(dim=-1, keepdim=True)
    ym = yw - yw.mean(dim=-1, keepdim=True)
    cov = (xm * ym).mean(dim=-1)
    std = xw.std(dim=-1) * yw.std(dim=-1) + 1e-8
    return cov / std


def _cs_rank(x: torch.Tensor) -> torch.Tensor:
    """截面排名：将每只股票在当日所有股票中的排名归一化到 [0, 1]。"""
    return x.argsort(dim=0).argsort(0).float() / (x.shape[0] + 1e-6)


@torch.jit.script
def _ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    if d == 0: return x
    pad = torch.zeros((x.shape[0], d), device=x.device)
    return torch.cat([pad, x[:, :-d]], dim=1)

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

OPS_CONFIG = [
    ('ADD', lambda x, y: x + y, 2),
    ('SUB', lambda x, y: x - y, 2),
    ('MUL', lambda x, y: x * y, 2),
    ('DIV', lambda x, y: x / (y + 1e-6), 2),
    ('NEG', lambda x: -x, 1),
    ('ABS', torch.abs, 1),
    ('SIGN', torch.sign, 1),
    ('GATE', _op_gate, 3),
    ('JUMP', _op_jump, 1),
    ('DECAY', _op_decay, 1),
    # 截面算子：对每个交易日做截面操作（dim=0 = 股票维度）
    ('CS_RANK', _cs_rank, 1),
    # 截面交互：排名乘积，捕捉多因子协同
    ('CROSS', lambda x, y: _cs_rank(x) * _cs_rank(y), 2),
    # 时序算子：固定窗口版本（VM 无法传参数，故预定义常用窗口）
    ('TS_RANK20', lambda x: _ts_rank(x, 20), 1),
    ('TS_MA20', lambda x: _ts_decay_linear(x, 20), 1),
    ('TS_DELTA5', lambda x: _ts_delta(x, 5), 1),
    ('TS_MIN5', lambda x: _ts_min(x, 5), 1),
    ('TS_MAX5', lambda x: _ts_max(x, 5), 1),
    ('TS_STD20', lambda x: _ts_std(x, 20), 1),
    ('TS_DELAY2', lambda x: _ts_delay(x, 2), 1),
    ('TS_CORR20', lambda x, y: _ts_corr(x, y, 20), 2),
]
