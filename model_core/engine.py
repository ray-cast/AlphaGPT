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
        """根据 action token 更新 open_slots 计数器（非in-place）。"""
        feat_count = len(self.model.features_list)
        is_op = action >= feat_count
        op_delta = self.arity_tens[action] - 1
        feat_delta = torch.full_like(action, -1)
        delta = torch.where(is_op, op_delta, feat_delta)
        delta[open_slots == 0] = 0
        open_slots = open_slots + delta  # 避免inplace操作
        return open_slots

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
        print(f"开始沪深300 Alpha Mining{lord_info} (REINFORCE)...")

        pbar = tqdm(range(ModelConfig.TRAIN_STEPS))

        # arity_map 用于公式去重
        feat_count = len(self.model.features_list)
        arity_map = {}
        for i, cfg in enumerate(OPS_CONFIG):
            arity_map[feat_count + i] = cfg[2]

        for step in pbar:
            current_max_len = ModelConfig.MAX_FORMULA_LEN

            bs = ModelConfig.BATCH_SIZE

            # 初始化进度条描述
            pbar.set_description(f"生成公式...")

            # --- Phase 1: Rollout (需要梯度用于log_probs) ---
            # 初始化（这些tensor不需要梯度）
            with torch.no_grad():
                open_slots = torch.ones(bs, dtype=torch.long, device=ModelConfig.DEVICE)
                bos_tokens = torch.full((bs, 1), self.model.bos_id, dtype=torch.long, device=ModelConfig.DEVICE)

            # 累积式序列生成：inp_buf 在循环中不断追加token，保持梯度连接
            inp_buf = bos_tokens  # [bs, 1]

            tokens_list = []
            log_probs = []
            values = []

            for t in range(current_max_len):
                # 前向传播（需要梯度）- 直接使用累积的inp_buf
                logits, value, _ = self.model(inp_buf)

                # Mask操作
                with torch.no_grad():
                    mask = self._get_strict_mask(open_slots, t, current_max_len)

                masked_logits = logits + mask
                dist = Categorical(logits=masked_logits)

                # 直接使用 policy 采样
                with torch.no_grad():
                    action = dist.sample()
                    open_slots = self._step_open_slots(open_slots, action)

                # 收集 critic value
                values.append(value.squeeze(-1))  # value维度是 [Batch, 1]，去掉最后一个维度

                # Log prob计算需要梯度
                action_log_prob = dist.log_prob(action)

                log_probs.append(action_log_prob)
                tokens_list.append(action)

                # 将新token追加到inp_buf，保持梯度连接
                inp_buf = torch.cat([inp_buf, action.unsqueeze(1)], dim=1)  # [bs, t+2]

            seqs = torch.stack(tokens_list, dim=1)

            rewards = torch.zeros(bs, device=ModelConfig.DEVICE)

            # 公式去重：相同公式只执行一次，结果映射回原位置
            formulas = seqs.tolist()

            unique_map = {}  # tuple(formula) -> (score, ret_val, alpha_values)
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
                    score, _, _ = unique_map[fkey]
                    rewards[i] = score
                    # 缓存命中时更新为缓存状态
                    pbar.set_description(f"缓存: {current_formula}")
                    continue

                res = self.vm.execute(
                    trimmed, self.loader.feat_tensor,
                    nan_mask=self.loader.nan_mask,
                    clean_feat=self.loader.clean_feat_tensor)

                if res is None:
                    unique_map[fkey] = (-1.0, 0.0, None)
                    rewards[i] = -1.0
                    continue

                if res.std() < 1e-4:
                    unique_map[fkey] = (-0.5, 0.0, res)
                    rewards[i] = -0.5
                    continue

                score, ret_val, _, sharpe = self.bt.evaluate(
                    res, self.loader.raw_data_cache, self.loader.target_ret,
                    start_idx=self.train_start, end_idx=self.train_end
                )

                # 保存单个策略信息
                unique_map[fkey] = (score.item(), ret_val, res)

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

            # --- Phase 2: Actor-Critic Update ---
            # 3. Update (REINFORCE with critic baseline)

            # 将values堆叠：[bs, max_len]
            stacked_values = torch.stack(values, dim=1)  # [bs, max_len]

            # Critic预测：使用所有时间步value的加权平均作为序列价值预测
            # 使用最后一个有效位置的value权重更大
            value_pred = stacked_values.mean(dim=1)  # [bs]

            # 计算advantage
            baseline = value_pred.detach()
            adv = rewards - baseline

            # 数值稳定性：裁剪 advantage 防止极端值
            adv = adv.clamp(-5.0, 5.0)

            # Policy loss：整个序列的log概率 * advantage（带discount factor）
            # log_probs: [max_len, bs]，需要转置为 [bs, max_len]
            stacked_log_probs = torch.stack(log_probs, dim=1)  # [bs, max_len]

            # 添加discount factor：早先生成的token权重更大（gamma > 1表示放大早期token的影响）
            # 对于"奖励只在最后"的情况，让早期token有更大权重更合理
            gamma = 1.0
            discount_weights = torch.tensor(
                [gamma ** (current_max_len - 1 - t) for t in range(current_max_len)],
                device=ModelConfig.DEVICE
            )  # [max_len]
            weighted_log_probs = (stacked_log_probs * discount_weights.unsqueeze(0)).sum(dim=1)  # [bs]
            policy_loss = -(weighted_log_probs * adv).mean()

            # Critic value loss: 让 critic 学习预测 reward
            value_loss = F.mse_loss(value_pred, rewards.detach())

            # Total loss
            loss = policy_loss + 0.5 * value_loss

            # 检查 loss 是否有效
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: Step {step} 检测到无效 loss，跳过更新")
                continue

            # 梯度更新（带梯度裁剪）
            self.opt.zero_grad()
            loss.backward()

            # 检查梯度是否包含 NaN/Inf
            grad_nan = False
            for param in self.model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        grad_nan = True
                        param.grad = torch.zeros_like(param.grad)

            if grad_nan:
                print(f"警告: Step {step} 检测到无效梯度，已清零")

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=ModelConfig.GRAD_CLIP_NORM)
            self.opt.step()

            if self.use_lord:
                self.lord_opt.step()

            self.scheduler.step()

            # 日志
            avg_reward = rewards.mean().item()
            postfix = {
                "AvgRew": f"{avg_reward:.3f}",
                "Best": f"{self.best_score:.3f}" if self.best_formula else "N/A",
                "Loss": f"{loss.item():.3f}",
                "Unique": f"{unique_ratio:.0%}",
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
            self.training_history["total_loss"].append(loss.item())
            self.training_history["policy_loss"].append(policy_loss.item())
            self.training_history["value_loss"].append(value_loss.item())
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
