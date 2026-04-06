import os
import json
import torch
import math
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

    def __init__(self, use_lord_regularization=True, lord_decay_rate=1e-3, lord_num_iterations=5,
                 seed_formulas=None, sft_steps=None):
        self.loader = AshareDataLoader()
        self.loader.load_data()

        self.model = AlphaGPT().to(ModelConfig.DEVICE)

        # 动态构建种子公式 token IDs
        self.seed_formulas = self._build_seed_formulas(seed_formulas)
        self.sft_steps = sft_steps if sft_steps is not None else ModelConfig.SFT_STEPS

        # Standard optimizer
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=3e-4)

        total_steps = ModelConfig.TRAIN_STEPS + self.sft_steps
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
        self.valid_idx = self.loader.valid_idx
        self.train_idx = self.loader.train_idx
        self.test_idx = self.loader.test_idx

        self.best_score = -float("inf")
        self.best_formula = None
        self.training_history = {
            "step": [],
            "avg_reward": [],
            "best_score": [],
            "stable_rank": [],
            "total_loss": [],
            "best_formula": None,
            "best_decoded": None,
        }

    def _build_seed_formulas(self, custom_seeds=None):
        """从名称动态构建种子公式的 token ID 列表。"""
        feat_name_to_id = {name: i for i, name in enumerate(self.model.features_list)}
        op_name_to_id = {cfg[0]: i + len(self.model.features_list) for i, cfg in enumerate(OPS_CONFIG)}

        seeds = []
        for op_name, feat_names in ModelConfig.SEED_FORMULA_NAMES:
            op_id = op_name_to_id.get(op_name)
            feat_ids = [feat_name_to_id.get(fn) for fn in feat_names]
            if op_id is None or any(fid is None for fid in feat_ids):
                continue
            seeds.append([op_id] + feat_ids)

        if custom_seeds:
            seeds.extend(custom_seeds)

        max_len = ModelConfig.MAX_FORMULA_LEN
        seeds = [s for s in seeds if len(s) <= max_len]
        return seeds

    def _get_curriculum_max_len(self, step):
        """根据课程学习计划返回当前步数的公式最大长度。"""
        schedule = ModelConfig.CURRICULUM_SCHEDULE
        if schedule is None:
            return ModelConfig.MAX_FORMULA_LEN
        max_len = ModelConfig.MAX_FORMULA_LEN
        for threshold, length in schedule:
            if step >= threshold:
                max_len = length
        return min(max_len, ModelConfig.MAX_FORMULA_LEN)

    def _get_strict_mask(self, open_slots, step, max_len=None):
        """严格 Action Masking：确保生成合法的前缀表达式树。"""
        if max_len is None:
            max_len = ModelConfig.MAX_FORMULA_LEN
        feat_count = len(self.model.features_list)
        vocab_size = self.model.vocab_size
        B = open_slots.shape[0]
        mask = torch.full((B, vocab_size), float('-inf'), device=ModelConfig.DEVICE)
        remaining_steps = max_len - step

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
        sft_info = f" (SFT {self.sft_steps} steps)" if self.seed_formulas else ""
        curriculum_info = " (Curriculum)" if ModelConfig.CURRICULUM_SCHEDULE else ""
        print(f"开始沪深300 Alpha Mining{lord_info}{sft_info}{curriculum_info}...")

        # ========== SFT 预训练：从已知好公式热启动 ==========
        if self.seed_formulas:
            self._run_sft()

        pbar = tqdm(range(ModelConfig.TRAIN_STEPS))

        # 预计算 arity 张量，用于向量化 open_slots 更新
        feat_count = len(self.model.features_list)
        vocab_size = self.model.vocab_size
        arity_tens = torch.zeros(vocab_size, dtype=torch.long, device=ModelConfig.DEVICE)
        arity_map = {}
        for i, cfg in enumerate(OPS_CONFIG):
            arity_tens[feat_count + i] = cfg[2]
            arity_map[feat_count + i] = cfg[2]

        prev_max_len = None
        for step in pbar:
            # 课程学习：获取当前最大公式长度
            current_max_len = self._get_curriculum_max_len(step)
            if prev_max_len is not None and current_max_len != prev_max_len:
                tqdm.write(f"[Curriculum] MAX_LEN {prev_max_len} → {current_max_len} at step {step}")
                self.patience_counter = 0  # 新搜索空间需要探索时间
            prev_max_len = current_max_len

            bs = ModelConfig.BATCH_SIZE
            inp = torch.zeros((bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)
            open_slots = torch.ones(bs, dtype=torch.long, device=ModelConfig.DEVICE)

            log_probs = []
            entropies = []
            values = []
            tokens_list = []
            open_slots_history = []
            for t in range(current_max_len):
                logits, val, _ = self.model(inp)
                mask = self._get_strict_mask(open_slots, t, current_max_len)
                dist = Categorical(logits=(logits + mask))
                action = dist.sample()

                log_probs.append(dist.log_prob(action))
                entropies.append(dist.entropy())
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
            formula_keys = [None] * bs  # 缓存 trimmed keys，避免重复计算
            for i in range(bs):
                vlen = self._valid_prefix_len(formulas[i], feat_count, arity_map)
                trimmed = formulas[i][:vlen]
                fkey = tuple(trimmed)
                formula_keys[i] = fkey
                if fkey in unique_map:
                    score, _ = unique_map[fkey]
                    rewards[i] = score
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

                score, ret_val = self.bt.evaluate(
                    res, self.loader.raw_data_cache, self.loader.target_ret,
                    start_idx=self.valid_idx, end_idx=self.train_idx
                )

                unique_map[fkey] = (score.item(), ret_val)
                rewards[i] = score

                if score.item() > self.best_score:
                    # 验证集检验（2017-2018 熊市压力测试）
                    val_score, _ = self.bt.evaluate(
                        res, self.loader.raw_data_cache, self.loader.target_ret,
                        start_idx=0, end_idx=self.valid_idx
                    )
                    if val_score < -1.0:
                        # 熊市验证集严重失效，跳过不更新 best
                        continue
                    self.best_score = score.item()
                    self.best_formula = trimmed
                    self.patience_counter = 0  # 重置早停计数器
                    decoded = self._decode(trimmed)
                    tqdm.write(
                        f"[!] New Best: Score {score:.2f} "
                        f"(Val={val_score:.2f}) "
                        f"CumRet {ret_val:.2%} | {decoded}"
                    )

            unique_ratio = len(unique_map) / bs

            # ---- Advantage 计算：top-k 筛选 + 归一化 ----
            # 训练进度（供 entropy 退火共用）
            progress = step / max(ModelConfig.TRAIN_STEPS - 1, 1)
            # Raw score 归一化
            rew_std = rewards.std() + 1e-6
            raw_normalized = (rewards - rewards.mean()) / rew_std
            raw_normalized = raw_normalized.clamp(-2, 2)

            # Top-k 筛选：动态比例，早期 30%→后期 10%
            topk_ratio = max(0.1, 0.3 - 0.2 * progress)
            k = max(int(bs * topk_ratio), 10)
            sorted_indices = rewards.argsort(descending=True)

            # top-k 内保留归一化分数，非 top-k 置零
            adv = torch.zeros(bs, device=ModelConfig.DEVICE)
            adv[sorted_indices[:k]] = raw_normalized[sorted_indices[:k]]

            # 多样性：重复惩罚
            diversity_bonus = torch.zeros(bs, device=ModelConfig.DEVICE)
            if unique_ratio < ModelConfig.DIVERSITY_TARGET:
                formula_counts = {}
                for i in range(bs):
                    fkey = formula_keys[i]
                    formula_counts[fkey] = formula_counts.get(fkey, 0) + 1
                for i in range(bs):
                    count = formula_counts[formula_keys[i]]
                    if count > 1:
                        diversity_bonus[i] = -ModelConfig.DIVERSITY_PENALTY * (count / bs)

            adv = adv + diversity_bonus

            # Critic baseline：用 value prediction 减去均值后的优势
            value_pred = torch.stack(values, 1).squeeze(-1).mean(dim=1)
            adv = adv - value_pred.detach()  # 用 critic 预测作为 baseline
            adv = adv - adv.mean()  # 去均值

            # Per-timestep advantage masking：只对 active steps 应用 advantage
            open_slots_at_t = torch.stack(open_slots_history, dim=1)  # [bs, MAX_FORMULA_LEN]

            policy_loss = 0
            entropy = 0
            for t in range(len(log_probs)):
                active = (open_slots_at_t[:, t] > 0).float()
                step_adv = adv * active
                policy_loss += -log_probs[t] * step_adv
                entropy += entropies[t].mean()
            policy_loss = policy_loss.mean()

            # Critic value loss（value_pred 已在上方计算）
            value_loss = F.mse_loss(value_pred, rewards.detach())

            # Entropy bonus：余弦退火，前期缓慢下降，后期加速衰减
            entropy_coef = ModelConfig.ENTROPY_COEF_END + 0.5 * (
                ModelConfig.ENTROPY_COEF_START - ModelConfig.ENTROPY_COEF_END
            ) * (1.0 + math.cos(math.pi * progress))
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
                "Len": current_max_len,
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

            with open("training_history.json", "w") as f:
                json.dump(self.training_history, f, ensure_ascii=False, indent=2)

            # 早停检查（确保课程学习有时间展开）
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

    def _run_sft(self):
        """监督预训练：让 Transformer 学会生成已知有效公式，作为 RL 热启动。"""
        all_seeds = self.seed_formulas
        max_len = ModelConfig.MAX_FORMULA_LEN
        all_seeds = [s for s in all_seeds if len(s) <= max_len]
        if not all_seeds:
            print("  [SFT] 无有效种子公式，跳过预训练")
            return

        print(f"  [SFT] 从 {len(all_seeds)} 个种子公式热启动...")
        for seed in all_seeds:
            print(f"    Seed: {self._decode(seed)}")

        sft_opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.model.train()

        feat_count = len(self.model.features_list)

        for step in range(self.sft_steps):
            sft_opt.zero_grad()

            # Round-robin：每步只训练一个公式，避免多公式共享初始状态导致的梯度冲突
            formula = all_seeds[step % len(all_seeds)]
            inp = torch.zeros((1, 1), dtype=torch.long, device=ModelConfig.DEVICE)
            loss = 0.0
            for t, target_token in enumerate(formula):
                logits, _, _ = self.model(inp)
                target = torch.tensor([target_token], device=ModelConfig.DEVICE)
                loss += F.cross_entropy(logits, target)
                inp = torch.cat([inp, target.unsqueeze(0)], dim=1)

            loss = loss / len(formula)
            loss.backward()
            sft_opt.step()

            if (step + 1) % 10 == 0:
                print(f"    [SFT] Step {step+1}/{self.sft_steps}, "
                      f"formula {self._decode(formula)}, loss: {loss.item():.4f}")

        # 验证：teacher-forced next-token 准确率
        # 注意：多个公式共享初始 token [0]，greedy decoding 只能复现一条序列，
        # 因此用 teacher-forced 逐 token 检查更合理。
        self.model.eval()
        with torch.no_grad():
            for formula in all_seeds:
                inp = torch.zeros((1, 1), dtype=torch.long, device=ModelConfig.DEVICE)
                correct = 0
                for t, target_token in enumerate(formula):
                    logits, _, _ = self.model(inp)
                    pred = logits.argmax(dim=-1).item()
                    if pred == target_token:
                        correct += 1
                    # Teacher forcing：使用正确 token 作为下一步输入
                    inp = torch.cat([inp, torch.tensor([[target_token]],
                                 device=ModelConfig.DEVICE)], dim=1)
                acc = correct / len(formula) * 100
                status = "OK" if correct == len(formula) else "PARTIAL"
                print(f"    [SFT-Verify] {status}: {self._decode(formula)} "
                      f"({correct}/{len(formula)} tokens, {acc:.0f}%)")

        self.model.train()

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
