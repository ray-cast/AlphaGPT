import torch
from .ops import OPS_CONFIG
from .factors import FeatureEngineer

class PrefixVM:
    def __init__(self):
        self.feat_offset = FeatureEngineer.INPUT_DIM
        self.op_map = {i + self.feat_offset: cfg[1] for i, cfg in enumerate(OPS_CONFIG)}
        self.arity_map = {i + self.feat_offset: cfg[2] for i, cfg in enumerate(OPS_CONFIG)}

    def execute(self, formula_tokens, feat_tensor, nan_mask=None, clean_feat=None):
        """递归下降求值前缀表达式。

        优化：nan_mask 和 clean_feat 由 data_loader 预计算，训练期间零开销。
        若未传入（generate_signals 等离线场景），回退到实时计算。
        """
        # 使用预计算版本或回退到实时计算
        if nan_mask is None:
            nan_mask = torch.isnan(feat_tensor).any(dim=1)
            clean_feat = feat_tensor.nan_to_num(nan=0.0)

        pos = [0]

        def eval_prefix():
            if pos[0] >= len(formula_tokens):
                return None
            token = int(formula_tokens[pos[0]])
            pos[0] += 1

            if token < self.feat_offset:
                return clean_feat[:, token, :]
            elif token in self.op_map:
                arity = self.arity_map[token]
                args = []
                for _ in range(arity):
                    arg = eval_prefix()
                    if arg is None:
                        return None
                    args.append(arg)
                func = self.op_map[token]
                res = func(*args)
                # 仅处理 Inf，NaN 不会出现（输入已清理）
                res = torch.nan_to_num(res, nan=0.0, posinf=1.0, neginf=-1.0)
                return res
            else:
                return None

        try:
            result = eval_prefix()
            if result is not None and pos[0] == len(formula_tokens):
                # 统一恢复停牌 NaN 标记
                result[nan_mask] = float('nan')
                return result
            return None
        except Exception:
            return None