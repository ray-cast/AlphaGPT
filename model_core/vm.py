import torch
from .ops import OPS_CONFIG
from .factors import FeatureEngineer

class PrefixVM:
    def __init__(self):
        self.feat_offset = FeatureEngineer.INPUT_DIM
        self.op_map = {i + self.feat_offset: cfg[1] for i, cfg in enumerate(OPS_CONFIG)}
        self.arity_map = {i + self.feat_offset: cfg[2] for i, cfg in enumerate(OPS_CONFIG)}

    def execute(self, formula_tokens, feat_tensor):
        """递归下降求值前缀表达式。
        前缀 RANK(DELTA5(RET)) = [RANK, DELTA5, RET]
          → 读 RANK(arity=1)，递归求1个参数
            → 读 DELTA5(arity=1)，递归求1个参数
              → 读 RET(特征)，返回数据
            → 对结果施加 DELTA5
          → 对结果施加 RANK
        """
        pos = [0]

        def eval_prefix():
            if pos[0] >= len(formula_tokens):
                return None
            token = int(formula_tokens[pos[0]])
            pos[0] += 1

            if token < self.feat_offset:
                return feat_tensor[:, token, :]
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
                if torch.isnan(res).any() or torch.isinf(res).any():
                    # 记录哪些位置是输入 NaN（停牌标记），清洗后再恢复
                    input_nan_mask = torch.zeros_like(res, dtype=torch.bool)
                    for a in args:
                        input_nan_mask |= torch.isnan(a)
                    res = torch.nan_to_num(res, nan=0.0, posinf=1.0, neginf=-1.0)
                    res[input_nan_mask] = float('nan')
                return res
            else:
                return None

        try:
            result = eval_prefix()
            # 所有 token 必须被消费（排除多余 token 的畸形表达式）
            if result is not None and pos[0] == len(formula_tokens):
                return result
            return None
        except Exception:
            return None