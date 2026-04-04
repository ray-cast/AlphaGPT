"""信号输出模块：将 alpha 分值转为可读的 CSV 信号文件。"""

import os
from datetime import datetime

import pandas as pd
import torch


class SignalWriter:
    def __init__(self, loader):
        """
        Args:
            loader: AshareDataLoader 实例（含 stock_codes, dates, split_idx）
        """
        self.loader = loader
        self.stock_codes = loader.stock_codes
        self.dates = loader.dates
        self.split_idx = loader.split_idx

    def write_signals(self, alpha_values: torch.Tensor, output_dir: str):
        """
        将 alpha 分值输出为 CSV 文件。

        Args:
            alpha_values: [num_stocks, T] tensor，来自 StackVM
            output_dir:   输出目录
        """
        os.makedirs(output_dir, exist_ok=True)

        # 只输出样本外日期
        oos_start = self.split_idx
        oos_dates = self.dates[oos_start:]
        alpha_oos = alpha_values[:, oos_start:]

        if alpha_oos.shape[1] != len(oos_dates):
            print(f"[WARN] alpha 维度 {alpha_oos.shape[1]} 与日期数 {len(oos_dates)} 不匹配")
            min_len = min(alpha_oos.shape[1], len(oos_dates))
            oos_dates = oos_dates[:min_len]
            alpha_oos = alpha_oos[:, :min_len]

        rows = []
        for t_idx, date in enumerate(oos_dates):
            col = alpha_oos[:, t_idx].cpu().numpy()
            # 排名（降序，分值越高排名越前）
            ranked_indices = col.argsort()[::-1]
            for rank, stock_idx in enumerate(ranked_indices, 1):
                score = float(col[stock_idx])
                direction = 1 if score > 0 else -1
                rows.append({
                    "date": date,
                    "ts_code": self.stock_codes[stock_idx],
                    "signal_score": round(score, 6),
                    "rank": rank,
                    "direction": direction,
                })

        df = pd.DataFrame(rows)

        # 写入单个 CSV
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(output_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        out_path = os.path.join(run_dir, "signals_all.csv")
        df.to_csv(out_path, index=False, encoding="utf-8-sig")

        # 额外输出：每天只保留 top30 的精简版
        df_top = df[df["rank"] <= 30]
        top_path = os.path.join(run_dir, "signals_top30.csv")
        df_top.to_csv(top_path, index=False, encoding="utf-8-sig")

        print(f"信号已写入:")
        print(f"  完整版: {out_path} ({len(df)} 行)")
        print(f"  Top30:  {top_path} ({len(df_top)} 行)")
        print(f"  日期范围: {oos_dates[0]} ~ {oos_dates[-1]}")
