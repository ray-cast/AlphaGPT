import os
import json
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

from .config import ModelConfig
from .data_loader import AshareDataLoader
from .model import NeuralSymbolicAlphaGenerator, NewtonSchulzLowRankDecay, StableRankMonitor
from .vm import PrefixVM
from .backtest import AshareBacktest
from .ops import OPS_CONFIG


class AlphaEngine:

    def __init__(self, use_lord_regularization=True, lord_decay_rate=1e-3, lord_num_iterations=5):
        self.loader = AshareDataLoader()
        self.loader.load_data()

        self.model = NeuralSymbolicAlphaGenerator().to(ModelConfig.DEVICE)

        # Standard optimizer
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        total_steps = ModelConfig.TRAIN_STEPS
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=total_steps, eta_min=1e-5
        )

        # LoRD 正则化
        self.use_lord = use_lord_regularization
        if self.use_lord:
            self.lord_opt = NewtonSchulzLowRankDecay(
                self.model.named_parameters(),
                decay_rate=lord_decay_rate,
                num_iterations=lord_num_iterations,
                target_keywords=["in_proj", "out_proj"]
            )
            self.rank_monitor = StableRankMonitor(
                self.model,
                target_keywords=["in_proj", "out_proj"]
            )
        else:
            self.lord_opt = None
            self.rank_monitor = None

        self.vm = PrefixVM()
        self.patience_counter = 0
        self.patience_limit = ModelConfig.PATIENCE_LIMIT
        self.bt = AshareBacktest()
        self.train_start = self.loader.train_start
        self.train_end = self.loader.train_end
        self.test_start = self.loader.test_start
        self.test_end = self.loader.test_end

        # 预计算 arity 张量，供 rollout 和 _evaluate_sequences 共用
        feat_count = len(self.model.features_list)
        self.arity_tens = torch.zeros(self.model.vocab_size, dtype=torch.long, device=ModelConfig.DEVICE)
        for i, cfg in enumerate(OPS_CONFIG):
            self.arity_tens[feat_count + i] = cfg[2]

        self.best_score = -999.0
        self.best_formula = None
        self.training_history = {
            "step": [],
            "avg_reward": [],
            "best_score": [],
            "stable_rank": [],
            "total_loss": [],
            "policy_loss": [],
            "value_loss": [],
            "best_formula": None,
            "best_decoded": None,
        }

    def _get_strict_mask(self, open_slots, step, max_len=None):
        """严格 Action Masking：确保生成合法的前缀表达式树。"""
        if max_len is None:
            max_len = ModelConfig.MAX_FORMULA_LEN
        vocab_size = self.model.vocab_size
        B = open_slots.shape[0]
        mask = torch.full((B, vocab_size), float('-inf'), device=ModelConfig.DEVICE)
        remaining_steps = max_len - step

        # 已完成（open_slots==0）→ pad 第一个 feature
        done_mask = (open_slots == 0)
        mask[done_mask, 0] = 0.0

        active_mask = ~done_mask
        # 每个 token 仅当 arity <= remaining_steps - open_slots 时允许
        # （该 token 消耗 1 步，其子树需 fill arity-1 个新 slot）
        budget = (remaining_steps - open_slots).clamp(min=0)        # [B]
        allowed = budget[:, None] >= self.arity_tens[None, :]       # [B, V]
        mask[allowed & active_mask[:, None]] = 0.0

        # 禁止在生成过程中输出 BOS token
        mask[:, self.model.bos_id] = float('-inf')
        return mask

    def _step_open_slots(self, open_slots, action):
        """根据 action token 更新 open_slots 计数器（in-place）。"""
        feat_count = len(self.model.features_list)
        is_op = action >= feat_count
        op_delta = self.arity_tens[action] - 1
        feat_delta = torch.full_like(action, -1)
        delta = torch.where(is_op, op_delta, feat_delta)
        delta[open_slots == 0] = 0
        open_slots += delta

    @staticmethod
    def _valid_prefix_len(tokens, feat_count, arity_map):
        """计算合法前缀表达式的实际长度（去除填充部分）。"""
        open_slots = 1
        for i, t in enumerate(tokens):
            t = int(t)
            if t < feat_count:
                open_slots -= 1
            elif t in arity_map:
                open_slots += arity_map[t] - 1
            else:
                return i
            if open_slots <= 0:
                return i + 1
        return len(tokens)

    def _evaluate_sequences(self, seqs, max_len):
        """Teacher-forcing: 用当前模型重新评估序列的 log_probs, values, entropy。"""
        B, T = seqs.shape
        open_slots = torch.ones(B, dtype=torch.long, device=ModelConfig.DEVICE)
        inp_buf = torch.full((B, T + 1), self.model.bos_id, dtype=torch.long, device=ModelConfig.DEVICE)

        log_probs, values, entropies = [], [], []

        for t in range(T):
            inp = inp_buf[:, :t + 1].clone()
            logits, val, _ = self.model(inp)
            mask = self._get_strict_mask(open_slots, t, max_len)
            dist = Categorical(logits=(logits + mask))
            action = seqs[:, t]

            log_probs.append(dist.log_prob(action))
            values.append(val)
            entropies.append(dist.entropy())

            inp_buf = inp_buf.clone()
            inp_buf[:, t + 1] = action
            self._step_open_slots(open_slots, action)

        return (
            torch.stack(log_probs, 1).sum(1),           # [B] total_log_prob
            torch.stack(values, 1).squeeze(-1).mean(1),  # [B] mean_value
            torch.stack(entropies, 1).sum(1),            # [B] total_entropy
        )

    def _compute_gae(self, rewards, values, gamma, lambda_gae):
        """
        计算Generalized Advantage Estimation (GAE)。
        平衡偏差和方差，比简单的adv = rewards - baseline更稳定。

        Args:
            rewards: [B] 每个序列的即时奖励
            values: [B] 每个序列的baseline值（critic输出）
            gamma: 折扣因子（通常0.99）
            lambda_gae: GAE参数（通常0.95，0=当前，1=累积）

        Returns:
            advantages: [B] 计算得到的advantages
        """
        advantages = torch.zeros_like(rewards)
        last_advantage = 0

        # 反向计算GAE
        for t in reversed(range(rewards.size(0))):
            if t == rewards.size(0) - 1:
                next_value = 0  # 最后一个状态没有未来
            else:
                next_value = values[t + 1]

            # TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
            delta = rewards[t] + gamma * next_value - values[t]

            # GAE累积: A_t = δ_t + γλA_{t+1}
            advantages[t] = last_advantage = delta + gamma * lambda_gae * last_advantage

        # 标准化advantages（保持数值稳定）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def train(self):
        lord_info = " (LoRD)" if self.use_lord else ""
        print(f"开始沪深300 Alpha Mining{lord_info}...")

        pbar = tqdm(range(ModelConfig.TRAIN_STEPS))

        # arity_map 用于公式去重
        feat_count = len(self.model.features_list)
        arity_map = {}
        for i, cfg in enumerate(OPS_CONFIG):
            arity_map[feat_count + i] = cfg[2]

        for step in pbar:
            current_max_len = ModelConfig.MAX_FORMULA_LEN

            bs = ModelConfig.BATCH_SIZE
            progress = step / max(ModelConfig.TRAIN_STEPS - 1, 1)

            # 初始化进度条描述
            pbar.set_description(f"生成公式...")

            # --- Phase 1: Rollout (no grad) ---
            eps = ModelConfig.EPS_GREEDY_START * (1 - progress) + ModelConfig.EPS_GREEDY_END * progress
            with torch.no_grad():
                open_slots = torch.ones(bs, dtype=torch.long, device=ModelConfig.DEVICE)
                inp_buf = torch.full((bs, current_max_len + 1), self.model.bos_id, dtype=torch.long, device=ModelConfig.DEVICE)
                tokens_list = []
                for t in range(current_max_len):
                    logits, _, _ = self.model(inp_buf[:, :t + 1])
                    mask = self._get_strict_mask(open_slots, t, current_max_len)
                    dist = Categorical(logits=(logits + mask))
                    # epsilon-greedy: 以 eps 概率在有效动作空间均匀采样
                    uniform = Categorical(probs=mask.softmax(dim=1).clamp(min=1e-8))
                    use_random = torch.rand(bs, device=ModelConfig.DEVICE) < eps
                    action = torch.where(use_random, uniform.sample(), dist.sample())
                    tokens_list.append(action)
                    inp_buf[:, t + 1] = action
                    self._step_open_slots(open_slots, action)

            seqs = torch.stack(tokens_list, dim=1)

            rewards = torch.zeros(bs, device=ModelConfig.DEVICE)

            # 公式去重：相同公式只执行一次，结果映射回原位置
            formulas = seqs.tolist()

            unique_map = {}  # tuple(formula) -> (score, ret_val)
            formula_keys = [None] * bs  # 缓存 trimmed keys，避免重复计算

            for i in range(bs):
                # 更新进度条描述显示当前正在计算的公式
                current_formula = self._decode(formulas[i])
                # 限制长度避免进度条过长
                if len(current_formula) > 30:
                    current_formula = current_formula[:27] + "..."

                # 显示当前正在计算的公式
                pbar.set_description(f"计算: {current_formula}")

                vlen = self._valid_prefix_len(formulas[i], feat_count, arity_map)
                trimmed = formulas[i][:vlen]
                fkey = tuple(trimmed)
                formula_keys[i] = fkey
                if fkey in unique_map:
                    score, _ = unique_map[fkey]
                    rewards[i] = score
                    # 缓存命中时更新为缓存状态
                    pbar.set_description(f"缓存: {current_formula}")
                    continue

                res = self.vm.execute(
                    trimmed, self.loader.feat_tensor,
                    nan_mask=self.loader.nan_mask,
                    clean_feat=self.loader.clean_feat_tensor)

                if res is None:
                    unique_map[fkey] = (-1.0, 0.0)
                    rewards[i] = -1.0
                    continue

                if res.std() < 1e-4:
                    unique_map[fkey] = (-0.5, 0.0)
                    rewards[i] = -0.5
                    continue

                score, ret_val, _, sharpe = self.bt.evaluate(
                    res, self.loader.raw_data_cache, self.loader.target_ret,
                    start_idx=self.train_start, end_idx=self.train_end
                )

                unique_map[fkey] = (score.item(), ret_val)
                rewards[i] = score

                if score.item() > self.best_score:
                    self.best_score = score.item()
                    self.best_formula = trimmed
                    self.patience_counter = 0  # 重置早停计数器
                    decoded = self._decode(trimmed)
                    tqdm.write(
                        f"[!] New Best: Score {score:.3f} "
                        f"CumRet {ret_val:.2%} "
                        f"Sharpe {sharpe:.2f} | {decoded}"
                    )

                    # 立即写盘保存最佳分数
                    self.training_history["best_formula"] = self.best_formula
                    self.training_history["best_decoded"] = decoded
                    self.training_history["best_score"].append(self.best_score)
                    with open("training_history.json", "w") as f:
                        json.dump(self.training_history, f, ensure_ascii=False, indent=2)

            unique_ratio = len(unique_map) / bs

            # 清理 NaN rewards（如 POW|CLOSE|CONST_-20 产生 inf/nan）
            nan_rew_count = torch.isnan(rewards).sum().item()
            if nan_rew_count > 0:
                rewards = rewards.nan_to_num(nan=-1.0)

            # ---- Phase 2: Old policy log_probs ----
            with torch.no_grad():
                old_log_probs, old_values, _ = self._evaluate_sequences(seqs, current_max_len)

            # ---- Phase 3: Advantage (GAE - Generalized Advantage Estimation) ----
            # 使用GAE替代简单的top-k + baseline方法
            # GAE的优势：
            # 1. 考虑累积效应（不仅是即时奖励）
            # 2. 利用全部样本（不再只取top-k）
            # 3. 平衡偏差和方差，提供更稳定的训练信号
            # 4. 不再强制中心化，保持真实的梯度信号
            adv = self._compute_gae(
                rewards.detach(),
                old_values.detach(),
                gamma=ModelConfig.GAMMA,
                lambda_gae=ModelConfig.GAE_LAMBDA
            )

            # Entropy 系数线性退火（比余弦退火探索期更长）
            entropy_coef = ModelConfig.ENTROPY_COEF_START * (1 - progress) + ModelConfig.ENTROPY_COEF_END * progress

            # ---- Phase 4: PPO Update ----
            total_loss_val = 0.0
            for _ in range(ModelConfig.PPO_EPOCHS):
                new_log_probs, new_values, new_entropy = self._evaluate_sequences(seqs, current_max_len)

                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - ModelConfig.PPO_CLIP_EPS,
                                    1 + ModelConfig.PPO_CLIP_EPS) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(new_values, rewards.detach())
                entropy_bonus = new_entropy.mean()

                loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy_bonus / current_max_len

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), ModelConfig.GRAD_CLIP_NORM)
                self.opt.step()

                if self.use_lord:
                    self.lord_opt.step()

                total_loss_val += loss.item()

            self.scheduler.step()

            # 日志
            avg_reward = rewards.mean().item()
            total_loss_val /= ModelConfig.PPO_EPOCHS
            postfix = {
                "AvgRew": f"{avg_reward:.3f}",
                "Best": f"{self.best_score:.3f}" if self.best_formula else "N/A",
                "Loss": f"{total_loss_val:.2f}",
                "Unique": f"{unique_ratio:.0%}",
                "Len": current_max_len,
            }

            if self.use_lord and step % 100 == 0:
                stable_rank = self.rank_monitor.compute()
                postfix["Rank"] = f"{stable_rank:.2f}"
                self.training_history["stable_rank"].append(stable_rank)

            self.training_history["step"].append(step)
            self.training_history["avg_reward"].append(avg_reward)
            self.training_history["best_score"].append(
                self.best_score if self.best_formula else None
            )
            self.training_history["total_loss"].append(total_loss_val)
            self.training_history["best_formula"] = self.best_formula
            self.training_history["best_decoded"] = self._decode(self.best_formula) if self.best_formula else None

            with open("training_history.json", "w") as f:
                json.dump(self.training_history, f, ensure_ascii=False, indent=2)

            # 早停检查
            self.patience_counter += 1
            if self.patience_counter >= self.patience_limit and step >= ModelConfig.MIN_TRAIN_STEPS:
                print(f"\n早停：连续 {self.patience_limit} 步无新最优，终止训练")
                break

            pbar.set_postfix(postfix)

        # 保存结果
        print(f"\n训练完成!")
        print(f"  最佳得分: {self.best_score:.4f}")
        print(f"  最佳公式: {self._decode(self.best_formula)}")

        # 保存 best_ashare_strategy.json 供 --signal-only 使用
        if self.best_formula is not None:
            import json as _json
            strategy_info = {
                "formula": self.best_formula,
                "decoded": self._decode(self.best_formula),
                "score": self.best_score,
            }
            with open("best_ashare_strategy.json", "w") as f:
                _json.dump(strategy_info, f, ensure_ascii=False, indent=2)
            print(f"  策略已保存至 best_ashare_strategy.json")

    def _decode(self, tokens):
        """将 token 序列解码为可读公式字符串。"""
        if tokens is None:
            return "None"
        feat_names = self.model.features_list
        op_names = [cfg[0] for cfg in __import__(
            "model_core.ops", fromlist=["OPS_CONFIG"]
        ).OPS_CONFIG]
        parts = []
        for t in tokens:
            if t < len(feat_names):
                parts.append(feat_names[t])
            elif t - len(feat_names) < len(op_names):
                parts.append(op_names[t - len(feat_names)])
            else:
                parts.append(f"?{t}")
        return " | ".join(parts)

    def generate_signals(self, output_dir: str = None):
        """训练后生成全量数据的买卖信号。"""
        if self.best_formula is None:
            print("尚未训练出有效公式，无法生成信号")
            return

        from .signal_writer import SignalWriter

        writer = SignalWriter(self.loader)
        alpha_values = self.vm.execute(
            self.best_formula, self.loader.feat_tensor,
            nan_mask=self.loader.nan_mask,
            clean_feat=self.loader.clean_feat_tensor
        )
        if alpha_values is None:
            print("最佳公式执行失败")
            return

        output_dir = output_dir or ModelConfig.SIGNAL_DIR
        writer.write_signals(alpha_values, output_dir)


if __name__ == "__main__":
    eng = AlphaEngine(use_lord_regularization=True)
    eng.train()
    eng.generate_signals()
