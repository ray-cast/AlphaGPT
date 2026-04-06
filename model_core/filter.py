"""基本面过滤与估值安全边际模块。

过滤规则:
    - PE_TTM > 0       隐含 EPS > 0（排除亏损股）
    - PE_TTM < MAX      排除高估值
    - ROE >= MIN        排除低效公司（ROE ≈ PB / PE_TTM）

安全边际（源自大盘鸡策略）:
    - P_VALUE = EPS_TTM × ROE × 100 → 股票内在价值
    - margin = ROE / PE_TTM × 100（等价于 P_VALUE / Price）
    - margin > 100% → 低估 → 截面加分
    - margin < 100% → 高估 → 截面减分
    - 使用 tanh 平滑过渡，避免硬切换

所有阈值在 config.py 中配置，数据缺失（NaN）时自动跳过，不误杀。
"""

import torch
from .config import ModelConfig


def apply_fundamental_filter(scores, pe_ttm=None, roe=None):
    """对 scores 矩阵应用基本面过滤，不合格股票置 -inf。

    Args:
        scores:  [N, T] alpha 分数矩阵（就地修改）
        pe_ttm:  [N, T] 滚动市盈率，None 则跳过 PE 过滤
        roe:     [N, T] 隐含净资产收益率，None 则跳过 ROE 过滤
    """
    if pe_ttm is not None:
        bad_pe = (pe_ttm <= 0) | (pe_ttm > ModelConfig.MAX_PE_TTM)
        bad_pe[torch.isnan(pe_ttm)] = False
        scores[bad_pe] = float('-inf')

    if roe is not None:
        bad_roe = roe < ModelConfig.MIN_ROE
        bad_roe[torch.isnan(roe) | torch.isinf(roe)] = False
        scores[bad_roe] = float('-inf')


def apply_valuation_margin(scores, pe_ttm=None, roe=None):
    """估值安全边际融入选股：低估股票截面加分，高估股票减分。

    核心逻辑源自大盘鸡策略：
        P_VALUE = EPS_TTM × ROE × 100 → 股票内在价值
        当 P_VALUE > 股价时低估（安全边际为正）→ 优先选入

    由于 P_VALUE / Price = ROE / PE_TTM × 100，
    可直接用 ROE / PE_TTM 衡量低估程度，无需价格数据。

    使用 tanh 平滑函数：margin=100% → boost=0，margin→∞ → boost→+weight，
    margin→0 → boost→-weight，避免硬切换。

    Args:
        scores:  [N, T] alpha 分数矩阵（就地修改）
        pe_ttm:  [N, T] 滚动市盈率，None 则跳过
        roe:     [N, T] 隐含净资产收益率，None 则跳过
    """
    weight = ModelConfig.VALUATION_MARGIN_WEIGHT
    if weight <= 0 or pe_ttm is None or roe is None:
        return

    # 安全边际指标 = ROE / PE_TTM × 100（等价于 P_VALUE / Price）
    # > 100% 表示 P_VALUE > 股价（低估）
    margin = roe / (pe_ttm + 1e-9) * 100.0

    # NaN / Inf 不影响原始分数
    bad = torch.isnan(margin) | torch.isinf(margin)
    margin = torch.where(bad, torch.zeros_like(margin), margin)

    # 已经被过滤（-inf）的股票不参与计算
    already_out = torch.isinf(scores) & (scores < 0)

    # 平滑 boost：tanh centered at margin=100%
    # margin=100% → tanh(0) = 0 → 无偏移
    # margin=150% → tanh(1.0) ≈ +0.76 → 低估加分
    # margin=50%  → tanh(-1.0) ≈ -0.76 → 高估减分
    boost = torch.tanh((margin - 1.0) * 2.0) * weight
    boost = torch.where(already_out | bad, torch.zeros_like(boost), boost)

    scores += boost
