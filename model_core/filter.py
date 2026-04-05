"""基本面过滤模块：基于 PE_TTM 和隐含 ROE 排除不合格股票。

过滤规则:
    - PE_TTM > 0       隐含 EPS > 0（排除亏损股）
    - PE_TTM < MAX      排除高估值
    - ROE >= MIN        排除低效公司（ROE ≈ PB / PE_TTM）

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
