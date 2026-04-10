import torch
import torch.nn.functional as F
from .config import ModelConfig


class AlphaEnsemble:
    """Alpha策略集成器，用于管理多个alpha策略并计算ensemble表现"""

    def __init__(self, max_size=10, ic_threshold=0.05):
        """
        Args:
            max_size: 保存的alpha策略最大数量
            ic_threshold: 策略最低IC阈值，低于此值的策略不保存
        """
        self.max_size = max_size
        self.ic_threshold = ic_threshold
        self.strategies = []  # 保存策略信息
        self.weights = None   # 策略权重

    def add_strategy(self, formula, score, cum_ret, ic, decoded=None, decoded_formula=None):
        """添加一个新的alpha策略到ensemble中"""
        decoded = decoded if decoded is not None else decoded_formula
        if decoded is None:
            raise ValueError("decoded or decoded_formula must be provided")
        strategy_info = {
            'formula': formula,
            'score': score,
            'cum_ret': cum_ret,
            'ic': ic,
            'decoded': decoded
        }

        # 检查是否已存在相同的策略
        exists = any(s['decoded'] == decoded for s in self.strategies)
        if exists:
            return False

        # 添加策略
        self.strategies.append(strategy_info)

        # 按IC降序排序
        self.strategies.sort(key=lambda x: x['ic'], reverse=True)

        # 保持最大数量限制
        if len(self.strategies) > self.max_size:
            self.strategies = self.strategies[:self.max_size]

        # 更新权重（基于IC和排名）
        self._update_weights()

        return True

    def _update_weights(self):
        """根据策略的IC和排名计算权重"""
        if not self.strategies:
            self.weights = None
            return

        # 使用IC的softmax作为权重基础
        ics = torch.tensor([s['ic'] for s in self.strategies])

        # 添加排名惩罚（排名靠后的策略权重较低）
        ranks = torch.arange(len(self.strategies), dtype=torch.float32)
        rank_penalty = 1.0 / (1.0 + ranks)  # 指数衰减

        # 组合权重
        combined_scores = ics * rank_penalty

        # softmax归一化
        weights = F.softmax(combined_scores, dim=0)
        self.weights = weights.tolist()

    def get_ensemble_factors(self, raw_factors_list, vm):
        """
        计算ensemble的alpha值
        Args:
            raw_factors_list: 每个策略的原始alpha值列表
            vm: PrefixVM实例用于计算alpha值
        Returns:
            ensemble_factors: ensemble的alpha值
        """
        if not self.strategies:
            return None

        # 计算所有策略的alpha值
        all_factors = []
        for strategy in self.strategies:
            formula = strategy['formula']
            alpha_values = vm.execute(
                formula, vm.feat_tensor,
                nan_mask=vm.nan_mask,
                clean_feat=vm.clean_feat_tensor
            )
            if alpha_values is not None:
                # 标准化alpha值（减去均值，除以标准差）
                alpha_values = (alpha_values - torch.mean(alpha_values)) / (torch.std(alpha_values) + 1e-8)
                all_factors.append(alpha_values)

        if not all_factors:
            return None

        # 加权组合
        all_factors = torch.stack(all_factors)  # [N, stocks, time]
        weights = torch.tensor(self.weights).view(-1, 1, 1)

        ensemble_factors = torch.sum(all_factors * weights, dim=0)
        return ensemble_factors

    def calculate_ensemble_ic(self, ensemble_factors, target_ret, valid_mask):
        """
        计算ensemble的IC
        """
        if ensemble_factors is None or self.weights is None:
            return 0.0

        # 使用与单个策略相同的IC计算函数
        return _vectorized_ic_raw(ensemble_factors, target_ret, valid_mask)

    def get_diversity_score(self):
        """计算ensemble的多样性得分（策略间的平均IC相关性）"""
        if len(self.strategies) < 2:
            return 1.0  # 单个策略默认为完全多样

        # 这里简化处理，实际可以计算策略间的相关性
        # 暂时使用策略IC的方差作为多样性指标
        ics = [s['ic'] for s in self.strategies]
        if len(ics) < 2:
            return 1.0

        ic_variance = torch.var(torch.tensor(ics)).item()
        # 方差越大，策略越多样化
        diversity = 1.0 / (1.0 + ic_variance)
        return diversity

    def save_ensemble(self, filepath):
        """保存ensemble策略到文件"""
        ensemble_data = {
            'strategies': self.strategies,
            'weights': self.weights,
            'diversity': self.get_diversity_score(),
            'avg_ic': torch.mean(torch.tensor([s['ic'] for s in self.strategies])).item() if self.strategies else 0.0
        }

        import json
        with open(filepath, 'w') as f:
            json.dump(ensemble_data, f, ensure_ascii=False, indent=2)

    def load_ensemble(self, filepath):
        """从文件加载ensemble策略"""
        import json
        with open(filepath, 'r') as f:
            ensemble_data = json.load(f)

        self.strategies = ensemble_data['strategies']
        self.weights = ensemble_data.get('weights')

    def summary(self):
        """返回ensemble摘要信息"""
        if not self.strategies:
            return "Ensemble is empty"

        summary = f"Ensemble ({len(self.strategies)} strategies):\n"
        summary += f"Average IC: {torch.mean(torch.tensor([s['ic'] for s in self.strategies])):.3f}\n"
        summary += f"Diversity Score: {self.get_diversity_score():.3f}\n"
        summary += f"Weights: {self.weights}\n\n"

        for i, strategy in enumerate(self.strategies[:5]):  # 显示前5个策略
            summary += f"Strategy {i+1}: IC={strategy['ic']:.3f}, Score={strategy['score']:.2f}, "
            summary += f"CumRet={strategy['cum_ret']:.2%}, Formula={strategy['decoded']}\n"

        return summary


@torch.jit.script
def _vectorized_ic_raw(factors: torch.Tensor, target_ret: torch.Tensor,
                       valid_mask: torch.Tensor) -> torch.Tensor:
    """向量化计算逐日 Pearson IC，返回原始均值。"""
    n_valid = valid_mask.float().sum(dim=0).clamp(min=1)

    f_c = torch.where(valid_mask, factors, 0.0)
    r_c = torch.where(valid_mask, target_ret, 0.0)
    f_mean = f_c.sum(dim=0) / n_valid
    r_mean = r_c.sum(dim=0) / n_valid
    f_c = torch.where(valid_mask, f_c - f_mean.unsqueeze(0), 0.0)
    r_c = torch.where(valid_mask, r_c - r_mean.unsqueeze(0), 0.0)

    cov = (f_c * r_c).sum(dim=0)
    std_f = (f_c.pow(2).sum(dim=0)).sqrt().clamp(min=1e-6)
    std_r = (r_c.pow(2).sum(dim=0)).sqrt().clamp(min=1e-6)
    daily_ic = cov / (std_f * std_r)

    valid_day = (n_valid >= 10) & (std_f > 1e-3) & (std_r > 1e-3)
    ic_count = valid_day.float().sum()
    if ic_count < 1:
        return torch.tensor(0.0, device=factors.device)

    avg_ic = (daily_ic * valid_day.float()).sum() / ic_count
    return avg_ic
