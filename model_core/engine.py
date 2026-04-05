import os
import json
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

from .config import ModelConfig
from .data_loader import AshareDataLoader
from .alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from .vm import PrefixVM
from .backtest import AshareBacktest
from .ops import OPS_CONFIG


class AlphaEngine:
    def __init__(self, use_lord_regularization=True, lord_decay_rate=1e-3, lord_num_iterations=5):
        self.loader = AshareDataLoader()
        self.loader.load_data()

        self.model = AlphaGPT().to(ModelConfig.DEVICE)

        # Standard optimizer
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=ModelConfig.TRAIN_STEPS, eta_min=1e-5
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

        self.best_score = -float("inf")
        self.best_formula = None
        self.seen_formulas = set()
        self.training_history = {
            "step": [],
            "avg_reward": [],
            "best_score": [],
            "stable_rank": [],
            "total_loss": [],
            "best_formula": None,
            "best_decoded": None,
        }

    def _get_strict_mask(self, open_slots, step):
        """严格 Action Masking：确保生成合法的前缀表达式树。"""
        feat_count = len(self.model.features_list)
        vocab_size = self.model.vocab_size
        B = open_slots.shape[0]
        mask = torch.full((B, vocab_size), float('-inf'), device=ModelConfig.DEVICE)
        remaining_steps = ModelConfig.MAX_FORMULA_LEN - step

        # 已完成（open_slots==0）→ pad 第一个 feature，保证序列合法
        done_mask = (open_slots == 0)
        mask[done_mask, 0] = 0.0

        active_mask = ~done_mask
        # 剩余步数不够填坑 → 必须选 feature（arity=0）
        must_pick_feat = (open_slots >= remaining_steps)

        mask[active_mask, :feat_count] = 0.0  # features 始终可选
        can_pick_op_mask = active_mask & (~must_pick_feat)
        if can_pick_op_mask.any():
            mask[can_pick_op_mask, feat_count:] = 0.0
        return mask

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

    def train(self):
        lord_info = " (LoRD)" if self.use_lord else ""
        print(f"开始沪深300 Alpha Mining{lord_info}...")

        pbar = tqdm(range(ModelConfig.TRAIN_STEPS))

        # 预计算 arity 张量，用于向量化 open_slots 更新
        feat_count = len(self.model.features_list)
        vocab_size = self.model.vocab_size
        arity_tens = torch.zeros(vocab_size, dtype=torch.long, device=ModelConfig.DEVICE)
        arity_map = {}
        for i, cfg in enumerate(OPS_CONFIG):
            arity_tens[feat_count + i] = cfg[2]
            arity_map[feat_count + i] = cfg[2]

        for step in pbar:
            bs = ModelConfig.BATCH_SIZE
            inp = torch.zeros((bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)
            open_slots = torch.ones(bs, dtype=torch.long, device=ModelConfig.DEVICE)

            log_probs = []
            values = []
            tokens_list = []
            open_slots_history = []
            warmup_active = step < ModelConfig.WARMUP_STEPS

            for t in range(ModelConfig.MAX_FORMULA_LEN):
                logits, val, _ = self.model(inp)
                mask = self._get_strict_mask(open_slots, t)
                if warmup_active:
                    # Warm-up 阶段：均匀采样（mask 仍生效保证合法性）
                    uniform_logits = torch.zeros_like(logits)
                    dist = Categorical(logits=(uniform_logits + mask))
                else:
                    dist = Categorical(logits=(logits + mask))
                action = dist.sample()

                log_probs.append(dist.log_prob(action))
                values.append(val)
                tokens_list.append(action)
                inp = torch.cat([inp, action.unsqueeze(1)], dim=1)

                # 更新 open_slots
                is_op = action >= feat_count
                op_delta = arity_tens[action] - 1
                feat_delta = torch.full_like(action, -1)
                delta = torch.where(is_op, op_delta, feat_delta)
                delta[open_slots == 0] = 0
                open_slots += delta
                open_slots_history.append(open_slots.clone())

            seqs = torch.stack(tokens_list, dim=1)

            rewards = torch.zeros(bs, device=ModelConfig.DEVICE)

            # 公式去重：相同公式只执行一次，结果映射回原位置
            formulas = seqs.tolist()
            unique_map = {}  # tuple(formula) -> (score, ret_val)
            for i in range(bs):
                vlen = self._valid_prefix_len(formulas[i], feat_count, arity_map)
                trimmed = formulas[i][:vlen]
                fkey = tuple(trimmed)
                if fkey in unique_map:
                    score, _ = unique_map[fkey]
                    rewards[i] = score
                    continue

                res = self.vm.execute(trimmed, self.loader.feat_tensor)

                if res is None:
                    unique_map[fkey] = (-5.0, 0.0)
                    rewards[i] = -5.0
                    continue

                if res.std() < 1e-4:
                    unique_map[fkey] = (-2.0, 0.0)
                    rewards[i] = -2.0
                    continue

                score, ret_val = self.bt.evaluate(
                    res, self.loader.raw_data_cache, self.loader.target_ret
                )

                # OOS 验证：混合 train + oos 得分防止过拟合
                oos_score = self.bt.evaluate_oos(
                    res, self.loader.raw_data_cache, self.loader.target_ret
                )
                combined_score = 0.6 * score + 0.4 * oos_score

                # 新颖性奖励：首次出现的公式获得额外 bonus
                if fkey not in self.seen_formulas:
                    combined_score = combined_score + ModelConfig.NOVELTY_BONUS
                    self.seen_formulas.add(fkey)

                unique_map[fkey] = (combined_score.item(), ret_val)
                rewards[i] = combined_score

                if combined_score.item() > self.best_score:
                    self.best_score = combined_score.item()
                    self.best_formula = trimmed
                    self.patience_counter = 0  # 重置早停计数器
                    decoded = self._decode(trimmed)
                    tqdm.write(
                        f"[!] New Best: Score {combined_score:.2f} "
                        f"(train={score:.2f}, oos={oos_score:.2f}) | "
                        f"CumRet {ret_val:.2%} | {decoded}"
                    )

            unique_ratio = len(unique_map) / bs

            # 混合奖励：rank + 归一化 raw score，避免公式同质化时梯度消失
            rank_indices = rewards.argsort().argsort().float()
            rank_normalized = rank_indices / (bs - 1) * 2 - 1  # 映射到 [-1, 1]

            # Raw score 归一化到 [-1, 1]
            rew_std = rewards.std() + 1e-6
            raw_normalized = (rewards - rewards.mean()) / rew_std
            raw_normalized = raw_normalized.clamp(-1, 1)

            adv = 0.7 * rank_normalized + 0.3 * raw_normalized

            # 公式长度 bonus：对数缩放，单 token 得 0，5 token 得 0.23
            formula_lengths = torch.ones(bs, device=ModelConfig.DEVICE)
            for i in range(bs):
                vlen = self._valid_prefix_len(formulas[i], feat_count, arity_map)
                formula_lengths[i] = max(vlen, 1)
            length_bonus = torch.log2(formula_lengths) * ModelConfig.LENGTH_BONUS_COEF

            # 多样性惩罚：对 batch 内重复出现的公式施加惩罚
            diversity_bonus = torch.zeros(bs, device=ModelConfig.DEVICE)
            if unique_ratio < ModelConfig.DIVERSITY_TARGET:
                formula_counts = {}
                for i in range(bs):
                    vlen = self._valid_prefix_len(formulas[i], feat_count, arity_map)
                    fkey = tuple(formulas[i][:vlen])
                    formula_counts[fkey] = formula_counts.get(fkey, 0) + 1
                for i in range(bs):
                    vlen = self._valid_prefix_len(formulas[i], feat_count, arity_map)
                    fkey = tuple(formulas[i][:vlen])
                    count = formula_counts[fkey]
                    if count > 1:
                        diversity_bonus[i] = -ModelConfig.DIVERSITY_PENALTY * (count / bs)

            adv = adv + length_bonus + diversity_bonus
            adv = adv - adv.mean()  # 去均值，确保 baselined

            # Per-timestep advantage masking：只对 active steps 应用 advantage
            open_slots_at_t = torch.stack(open_slots_history, dim=1)  # [bs, MAX_FORMULA_LEN]

            policy_loss = 0
            entropy = 0
            for t in range(len(log_probs)):
                active = (open_slots_at_t[:, t] > 0).float()
                step_adv = adv * active
                policy_loss += -log_probs[t] * step_adv
                entropy += -(log_probs[t].exp() * log_probs[t]).sum()
            policy_loss = policy_loss.mean()

            # Critic value loss
            value_pred = torch.stack(values, 1).squeeze(-1).mean(dim=1)
            value_loss = F.mse_loss(value_pred, rewards.detach())

            # Entropy bonus：线性退火，去掉 / bs 使 entropy 系数真正有效
            progress = step / max(ModelConfig.TRAIN_STEPS - 1, 1)
            entropy_coef = ModelConfig.ENTROPY_COEF_START * (1.0 - progress) + ModelConfig.ENTROPY_COEF_END * progress
            loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy / len(log_probs)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            if self.use_lord:
                self.lord_opt.step()

            self.scheduler.step()

            # 日志
            avg_reward = rewards.mean().item()
            total_loss_val = loss.item()
            postfix = {
                "AvgRew": f"{avg_reward:.3f}",
                "Best": f"{self.best_score:.3f}",
                "Loss": f"{total_loss_val:.2f}",
                "Unique": f"{unique_ratio:.0%}",
            }

            if self.use_lord and step % 100 == 0:
                stable_rank = self.rank_monitor.compute()
                postfix["Rank"] = f"{stable_rank:.2f}"
                self.training_history["stable_rank"].append(stable_rank)

            self.training_history["step"].append(step)
            self.training_history["avg_reward"].append(avg_reward)
            self.training_history["best_score"].append(self.best_score)
            self.training_history["total_loss"].append(total_loss_val)
            self.training_history["best_formula"] = self.best_formula
            self.training_history["best_decoded"] = self._decode(self.best_formula) if self.best_formula else None

            # 每 10 步或最后一步写盘，减少 I/O 开销
            if step % 10 == 0 or step == ModelConfig.TRAIN_STEPS - 1:
                with open("training_history.json", "w") as f:
                    json.dump(self.training_history, f, ensure_ascii=False, indent=2)

            # 早停检查
            self.patience_counter += 1
            if self.patience_counter >= self.patience_limit:
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
            self.best_formula, self.loader.feat_tensor
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
