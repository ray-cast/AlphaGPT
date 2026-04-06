"""基本面过滤模块。

过滤规则:
    - PE_TTM > 0       隐含 EPS > 0（排除亏损股）
    - PE_TTM < MAX      排除高估值
    - ROE >= MIN        排除低效公司（ROE ≈ PB / PE_TTM）

所有阈值在 config.py 中配置，数据缺失（NaN）时自动跳过，不误杀。
"""

import torch
from .config import ModelConfig


def apply_fundamental_filter(scores, pe_ttm=None, roe=None, soft=False):
    """对 scores 矩阵应用基本面过滤。

    Args:
        scores:  [N, T] alpha 分数矩阵（就地修改）
        pe_ttm:  [N, T] 滚动市盈率，None 则跳过 PE 过滤
        roe:     [N, T] 隐含净资产收益率，None 则跳过 ROE 过滤
        soft:    True 时用 soft penalty 替代 -inf 硬截断（训练用）
    """
    if pe_ttm is not None:
        bad_pe = (pe_ttm <= 0) | (pe_ttm > ModelConfig.MAX_PE_TTM)
        bad_pe[torch.isnan(pe_ttm)] = False
        if soft:
            scores[bad_pe] -= 2.0
        else:
            scores[bad_pe] = float('-inf')

    if roe is not None:
        bad_roe = roe < ModelConfig.MIN_ROE
        bad_roe[torch.isnan(roe) | torch.isinf(roe)] = False
        if soft:
            scores[bad_roe] -= 2.0
        else:
            scores[bad_roe] = float('-inf')
