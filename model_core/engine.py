import os
import json
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

from .config import ModelConfig
from .data_loader import AshareDataLoader
from .alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from .vm import StackVM
from .backtest import AshareBacktest


class AlphaEngine:
    def __init__(self, use_lord_regularization=True, lord_decay_rate=1e-3, lord_num_iterations=5):
        self.loader = AshareDataLoader()
        self.loader.load_data()

        self.model = AlphaGPT().to(ModelConfig.DEVICE)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        # LoRD 正则化
        self.use_lord = use_lord_regularization
        if self.use_lord:
            self.lord_opt = NewtonSchulzLowRankDecay(
                self.model.named_parameters(),
                decay_rate=lord_decay_rate,
                num_iterations=lord_num_iterations,
                target_keywords=["q_proj", "k_proj", "attention", "qk_norm"]
            )
            self.rank_monitor = StableRankMonitor(
                self.model,
                target_keywords=["q_proj", "k_proj"]
            )
        else:
            self.lord_opt = None
            self.rank_monitor = None

        self.vm = StackVM()
        self.bt = AshareBacktest()

        self.best_score = -float("inf")
        self.best_formula = None
        self.training_history = {
            "step": [],
            "avg_reward": [],
            "best_score": [],
            "stable_rank": []
        }

    def train(self):
        lord_info = " (LoRD)" if self.use_lord else ""
        print(f"开始沪深300 Alpha Mining{lord_info}...")

        pbar = tqdm(range(ModelConfig.TRAIN_STEPS))

        for step in pbar:
            bs = ModelConfig.BATCH_SIZE
            inp = torch.zeros((bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)

            log_probs = []
            values = []
            tokens_list = []

            for _ in range(ModelConfig.MAX_FORMULA_LEN):
                logits, val, _ = self.model(inp)
                dist = Categorical(logits=logits)
                action = dist.sample()

                log_probs.append(dist.log_prob(action))
                values.append(val)
                tokens_list.append(action)
                inp = torch.cat([inp, action.unsqueeze(1)], dim=1)

            seqs = torch.stack(tokens_list, dim=1)

            rewards = torch.zeros(bs, device=ModelConfig.DEVICE)

            for i in range(bs):
                formula = seqs[i].tolist()

                res = self.vm.execute(formula, self.loader.feat_tensor)

                if res is None:
                    rewards[i] = -5.0
                    continue

                if res.std() < 1e-4:
                    rewards[i] = -2.0
                    continue

                score, ret_val = self.bt.evaluate(
                    res, self.loader.raw_data_cache, self.loader.target_ret
                )
                rewards[i] = score

                if score.item() > self.best_score:
                    self.best_score = score.item()
                    self.best_formula = formula
                    decoded = self._decode(formula)
                    tqdm.write(
                        f"[!] New Best: Score {score:.2f} | "
                        f"CumRet {ret_val:.2%} | {decoded}"
                    )

            # REINFORCE with critic baseline
            baseline = torch.stack(values, 1).squeeze(-1).mean(dim=1).detach()
            adv = (rewards - baseline) / (rewards.std() + 1e-5)
            policy_loss = 0
            for t in range(len(log_probs)):
                policy_loss += -log_probs[t] * adv
            policy_loss = policy_loss.mean()
            # Critic value loss
            value_pred = torch.stack(values, 1).squeeze(-1).mean(dim=1)
            value_loss = F.mse_loss(value_pred, rewards.detach())
            loss = policy_loss + 0.5 * value_loss

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            if self.use_lord:
                self.lord_opt.step()

            # 日志
            avg_reward = rewards.mean().item()
            postfix = {"AvgRew": f"{avg_reward:.3f}", "Best": f"{self.best_score:.3f}"}

            if self.use_lord and step % 100 == 0:
                stable_rank = self.rank_monitor.compute()
                postfix["Rank"] = f"{stable_rank:.2f}"
                self.training_history["stable_rank"].append(stable_rank)

            self.training_history["step"].append(step)
            self.training_history["avg_reward"].append(avg_reward)
            self.training_history["best_score"].append(self.best_score)

            pbar.set_postfix(postfix)

        # 保存结果
        with open("best_ashare_strategy.json", "w") as f:
            json.dump({
                "formula": self.best_formula,
                "score": self.best_score,
                "decoded": self._decode(self.best_formula) if self.best_formula else None,
            }, f, ensure_ascii=False, indent=2)

        with open("training_history.json", "w") as f:
            json.dump(self.training_history, f, ensure_ascii=False, indent=2)

        print(f"\n训练完成!")
        print(f"  最佳得分: {self.best_score:.4f}")
        print(f"  最佳公式: {self._decode(self.best_formula)}")

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
