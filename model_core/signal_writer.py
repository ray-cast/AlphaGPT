"""信号输出模块：将 alpha 分值转为可读的 CSV 信号文件。"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch


class SignalWriter:
    def __init__(self, loader):
        """
        Args:
            loader: AshareDataLoader 实例（含 stock_codes, dates, split_idx, raw_data_cache）
        """
        self.loader = loader
        self.stock_codes = loader.stock_codes
        self.dates = loader.dates
        self.split_idx = loader.split_idx

        # 预计算市场趋势指标（close / MA60 - 1，经 tanh 压缩）
        close = loader.raw_data_cache["close"]  # [num_stocks, T]
        market_close = close.mean(dim=0)  # 等权均价作为市场代理
        T = market_close.shape[0]
        ma60 = torch.zeros_like(market_close)
        if T >= 60:
            cumsum = torch.cumsum(market_close, dim=0)
            ma60[59] = cumsum[59] / 60.0
            ma60[60:] = (cumsum[60:] - cumsum[:-60]) / 60.0
        valid = ma60 > 0
        trend = torch.zeros_like(market_close)
        trend[valid] = (market_close[valid] / ma60[valid] - 1)
        self.market_trend = torch.tanh(trend * 5.0)  # 压缩到 [-1, 1]

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
        trend_oos = self.market_trend[oos_start:]

        if alpha_oos.shape[1] != len(oos_dates):
            print(f"[WARN] alpha 维度 {alpha_oos.shape[1]} 与日期数 {len(oos_dates)} 不匹配")
            min_len = min(alpha_oos.shape[1], len(oos_dates))
            oos_dates = oos_dates[:min_len]
            alpha_oos = alpha_oos[:, :min_len]
            trend_oos = trend_oos[:min_len]

        rows = []
        for t_idx, date in enumerate(oos_dates):
            col = alpha_oos[:, t_idx].cpu().numpy()
            trend_score = float(trend_oos[t_idx].cpu())

            # 截面中位数作为多空分界线
            median_score = np.median(col)

            # 排名（降序，分值越高排名越前）
            ranked_indices = col.argsort()[::-1]
            for rank, stock_idx in enumerate(ranked_indices, 1):
                score = float(col[stock_idx])
                # 基于截面中位数区分方向：高于中位数为做多(1)，否则为看空(-1)
                direction = 1 if score > median_score else -1
                # 市场趋势向下时，即使截面看多也降低权重
                if trend_score < -0.3 and direction == 1:
                    direction = 0  # 观望
                rows.append({
                    "date": date,
                    "ts_code": self.stock_codes[stock_idx],
                    "signal_score": round(score, 6),
                    "rank": rank,
                    "direction": direction,
                    "market_trend": round(trend_score, 4),
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
