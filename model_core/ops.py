import torch
import torch.jit


def _ts_delta(x: torch.Tensor, d: int) -> torch.Tensor:
    """时序变化量：x(t) - x(t-d)。"""
    return x - _ts_delay(x, d)


def _ts_zscore(x: torch.Tensor, d: int) -> torch.Tensor:
    """时序标准化：(x - MA_d) / STD_d。"""
    if d <= 1: return torch.zeros_like(x)
    B, T = x.shape
    pad = torch.zeros((B, d - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    mean = windows.mean(dim=-1)
    std = windows.std(dim=-1) + 1e-6
    return (x - mean) / std


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


def _cs_rank(x: torch.Tensor) -> torch.Tensor:
    """截面排名：将每只股票在当日所有股票中的排名归一化到 [0, 1]。"""
    return x.argsort(dim=0).argsort(0).float() / (x.shape[0] + 1e-6)


def _cs_zscore(x: torch.Tensor) -> torch.Tensor:
    """截面标准化：对每个交易日做 z-score。"""
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True) + 1e-6
    return (x - mean) / std

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
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True) + 1e-6
    z = (x - mean) / std
    return torch.relu(z - 3.0)

@torch.jit.script
def _op_decay(x: torch.Tensor) -> torch.Tensor:
    return x + 0.8 * _ts_delay(x, 1) + 0.6 * _ts_delay(x, 2)

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
    ('DELAY1', lambda x: _ts_delay(x, 1), 1),
    ('MAX3', lambda x: torch.max(x, torch.max(_ts_delay(x,1), _ts_delay(x,2))), 1),

    # 时序算子 — 来自 times.py，补齐多日窗口时序统计能力
    ('DELTA5',    lambda x: _ts_delta(x, 5),           1),  # 5日变化量
    ('MA20',      lambda x: _ts_decay_linear(x, 20),   1),  # 20日线性衰减均线
    ('STD20',     lambda x: _ts_zscore(x, 20),         1),  # 20日时序标准化
    ('TS_RANK20', lambda x: _ts_rank(x, 20),           1),  # 20日时序排名百分位

    # 截面算子 — 量化选股核心操作
    ('RANK',   _cs_rank,   1),    # 截面排名归一化
    ('ZSCORE', _cs_zscore, 1),    # 截面标准化

    # 时序-截面混合算子 — 单 token 实现"先时序后截面"，大幅降低搜索难度
    ('RANK_DELTA5',  lambda x: _cs_rank(x - _ts_delay(x, 5)),        1),  # 截面排名(5日动量)
    ('RANK_TS_RANK', lambda x: _cs_rank(_ts_rank(x, 20)),            1),  # 截面排名(20日时序百分位)
    ('ZSCORE_STD20', lambda x: _cs_zscore(_ts_zscore(x, 20)),        1),  # 截面标准化(20日时序标准化)
    ('RANK_MA20',    lambda x: _cs_rank(_ts_decay_linear(x, 20)),    1),  # 截面排名(20日衰减均线)
]

# 算子分类索引（相对于 OPS_CONFIG 的偏移量），用于 engine.py 的混合结构奖励
_TS_OPS = {'DELTA5', 'MA20', 'STD20', 'TS_RANK20', 'DELAY1', 'DECAY', 'MAX3'}
_CS_OPS = {'RANK', 'ZSCORE'}
TS_OP_INDICES = {i for i, cfg in enumerate(OPS_CONFIG) if cfg[0] in _TS_OPS}
CS_OP_INDICES = {i for i, cfg in enumerate(OPS_CONFIG) if cfg[0] in _CS_OPS}