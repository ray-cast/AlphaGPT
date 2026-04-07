"""信号输出模块：将 alpha 分值转为可读的 CSV 信号文件。"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from .config import ModelConfig


class SignalWriter:
    def __init__(self, loader):
        """
        Args:
            loader: AshareDataLoader 实例（含 stock_codes, dates, test_start, raw_data_cache）
        """
        self.loader = loader
        self.stock_codes = loader.stock_codes
        self.dates = loader.dates
        self.test_start = loader.test_start
        self.test_end = loader.test_end

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
            alpha_values: [num_stocks, T] tensor，来自 PrefixVM
            output_dir:   输出目录
        """
        os.makedirs(output_dir, exist_ok=True)

        # 输出测试集及以后的信号
        val_start = self.test_start
        val_dates = self.dates[val_start:]
        alpha_val = alpha_values[:, val_start:]
        trend_val = self.market_trend[val_start:]

        if alpha_val.shape[1] != len(val_dates):
            print(f"[WARN] alpha 维度 {alpha_val.shape[1]} 与日期数 {len(val_dates)} 不匹配")
            min_len = min(alpha_val.shape[1], len(val_dates))
            val_dates = val_dates[:min_len]
            alpha_val = alpha_val[:, :min_len]
            trend_val = trend_val[:min_len]

        rows = []
        for t_idx, date in enumerate(val_dates):
            col = alpha_val[:, t_idx].cpu().numpy()
            trend_score = float(trend_val[t_idx].cpu())

            # 排除 NaN 后再排名（降序，分值越高排名越前）
            valid_mask = ~np.isnan(col)
            ranked_indices_all = np.full(len(col), -1, dtype=int)
            if valid_mask.any():
                valid_idx = np.where(valid_mask)[0]
                sorted_order = col[valid_idx].argsort()[::-1]
                ranked_indices_all[:len(valid_idx)] = valid_idx[sorted_order]
            # NaN 股票排到最后
            nan_idx = np.where(~valid_mask)[0]
            ranked_indices_all[len(valid_mask) - len(nan_idx):] = nan_idx

            # 截面中位数（排除 NaN）
            valid_scores = col[valid_mask]
            median_score = np.median(valid_scores) if len(valid_scores) > 0 else 0.0

            for rank, stock_idx in enumerate(ranked_indices_all, 1):
                score = float(col[stock_idx])
                if np.isnan(score):
                    direction = -1
                else:
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

        # 额外输出：每天只保留 top N 的精简版
        top_n = ModelConfig.TOP_N_STOCKS
        df_top = df[df["rank"] <= top_n]
        top_path = os.path.join(run_dir, f"signals_top{top_n}.csv")
        df_top.to_csv(top_path, index=False, encoding="utf-8-sig")

        print(f"信号已写入:")
        print(f"  完整版: {out_path} ({len(df)} 行)")
        print(f"  Top{top_n}:  {top_path} ({len(df_top)} 行)")
        print(f"  日期范围: {val_dates[0]} ~ {val_dates[-1]}")
