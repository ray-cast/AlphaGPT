import torch
from .ops import OPS_CONFIG
from .factors import FeatureEngineer

class PrefixVM:
    def __init__(self):
        self.feat_offset = FeatureEngineer.INPUT_DIM
        self.op_map = {i + self.feat_offset: cfg[1] for i, cfg in enumerate(OPS_CONFIG)}
        self.arity_map = {i + self.feat_offset: cfg[2] for i, cfg in enumerate(OPS_CONFIG)}
        self._nan_mask_cache = None
        self._clean_feat_cache = None
        self._cache_key = None

    def _prepare_tensors(self, feat_tensor):
        """预计算 NaN mask 和清理版 feat_tensor，避免逐算子 NaN 处理。"""
        key = feat_tensor.data_ptr()
        if self._cache_key == key:
            return self._nan_mask_cache, self._clean_feat_cache
        # 任意 feature 中有 NaN 的位置（停牌股票）
        self._nan_mask_cache = torch.isnan(feat_tensor).any(dim=1)
        # NaN 替换为 0 的干净版本
        self._clean_feat_cache = feat_tensor.nan_to_num(nan=0.0)
        self._cache_key = key
        return self._nan_mask_cache, self._clean_feat_cache

    def execute(self, formula_tokens, feat_tensor):
        """递归下降求值前缀表达式。
        前缀 RANK(DELTA5(RET)) = [RANK, DELTA5, RET]
          → 读 RANK(arity=1)，递归求1个参数
            → 读 DELTA5(arity=1)，递归求1个参数
              → 读 RET(特征)，返回数据
            → 对结果施加 DELTA5
          → 对结果施加 RANK

        优化：预清理 feat_tensor 中的 NaN，执行完后统一恢复停牌标记，
        避免每个算子都做 NaN 检测和恢复。
        """
        nan_mask, clean_feat = self._prepare_tensors(feat_tensor)
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