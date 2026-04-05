"""样本外业绩评估模块：输出年化收益、夏普、最大回撤等指标，并绘制净值曲线。"""

import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch

from .config import ModelConfig


class StrategyReport:
    def __init__(self, loader):
        self.loader = loader
        self.top_n = ModelConfig.TOP_N_STOCKS
        self.buy_cost = ModelConfig.TOTAL_BUY_COST
        self.sell_cost = ModelConfig.TOTAL_SELL_COST
        self.min_turnover = ModelConfig.MIN_TURNOVER_RATE

    def evaluate(self, alpha_values):
        """
        在 OOS 区间评估策略表现。

        Args:
            alpha_values: [num_stocks, T] 来自 PrefixVM 的 alpha 分值
        Returns:
            metrics: dict，含 ann_ret / ann_vol / sharpe / max_dd / calmar / turnover
            daily_ret: np.ndarray [T_oos] 每日组合收益
            bench_ret: np.ndarray [T_oos] 每日基准收益
        """
        loader = self.loader
        split = loader.split_idx
        target_ret = loader.target_ret  # [num_stocks, T]
        turnover_rate = loader.raw_data_cache.get(
            "turnover_rate", torch.zeros_like(alpha_values)
        )

        # OOS 区间
        alpha_oos = alpha_values[:, split:]
        target_oos = target_ret[:, split:]
        turnover_oos = turnover_rate[:, split:]
        oos_dates = loader.dates[split:]

        N, T = alpha_oos.shape

        # 排除停牌（NaN 信号或 NaN 收益）
        valid_mask = ~(torch.isnan(alpha_oos) | torch.isnan(target_oos))

        # 构建持仓（复用 AshareBacktest 的逻辑，含动态仓位管理）
        valid_target = target_oos.clone()
        valid_target[~valid_mask] = 0.0
        valid_count = valid_mask.float().sum(dim=0).clamp(min=1)
        market_daily_ret = valid_target.sum(dim=0) / valid_count
        market_ma20 = self._rolling_mean_1d(market_daily_ret, 20)

        position = torch.zeros_like(alpha_oos)
        for t in range(T):
            alpha_t = alpha_oos[:, t].clone()
            # 排除停牌和低换手
            valid_t = valid_mask[:, t] & (turnover_oos[:, t] > self.min_turnover)
            alpha_t[~valid_t] = -float("inf")
            k = min(self.top_n, valid_t.sum().int().item())
            if market_ma20[t] < 0:
                k = max(k // 2, 0)
            if k > 0:
                _, topk_idx = torch.topk(alpha_t, k)
                position[topk_idx, t] = 1.0

        # 换手
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0.0
        turnover = torch.abs(position - prev_pos)
        avg_cost = (self.buy_cost + self.sell_cost) / 2.0
        tx_cost = turnover * avg_cost

        # 组合收益（绝对收益）
        gross_pnl = position * target_oos
        net_pnl = gross_pnl - tx_cost

        daily_ret = net_pnl.sum(dim=0).cpu().numpy() / (self.top_n + 1e-9)
        bench_daily = market_daily_ret.cpu().numpy()  # 已排除停牌的等权市场均值

        # 统计
        metrics = self._compute_metrics(daily_ret, bench_daily, turnover, T)
        return metrics, daily_ret, bench_daily, oos_dates

    def _compute_metrics(self, daily_ret, bench_daily, turnover, T):
        equity = np.cumprod(1 + daily_ret)
        bench_equity = np.cumprod(1 + bench_daily)

        total_ret = equity[-1] - 1
        ann_ret = equity[-1] ** (252 / T) - 1 if T > 0 else 0.0
        ann_vol = np.std(daily_ret) * np.sqrt(252)
        sharpe = (ann_ret - 0.02) / (ann_vol + 1e-6)

        # 最大回撤
        dd = 1 - equity / np.maximum.accumulate(equity)
        max_dd = np.max(dd)
        calmar = ann_ret / (max_dd + 1e-6)

        # 基准统计
        bench_total = bench_equity[-1] - 1
        bench_ann = bench_equity[-1] ** (252 / T) - 1 if T > 0 else 0.0

        # 超额统计
        excess_daily = daily_ret - bench_daily
        excess_equity = np.cumprod(1 + excess_daily)
        excess_total = excess_equity[-1] - 1

        avg_turnover = turnover.mean().item()

        # 胜率
        win_rate = np.mean(daily_ret > 0)

        return {
            "total_ret": total_ret,
            "ann_ret": ann_ret,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "calmar": calmar,
            "bench_total": bench_total,
            "bench_ann": bench_ann,
            "excess_total": excess_total,
            "avg_turnover": avg_turnover,
            "win_rate": win_rate,
            "trading_days": T,
        }

    def print_report(self, metrics, oos_dates):
        print("\n" + "=" * 60)
        print("  Out-of-Sample Performance Report")
        print("=" * 60)
        print(f"  Test Period     : {oos_dates[0]} ~ {oos_dates[-1]}")
        print(f"  Trading Days    : {metrics['trading_days']}")
        print("-" * 60)
        print(f"  Strategy Total  : {metrics['total_ret']:+.2%}")
        print(f"  Strategy Ann.   : {metrics['ann_ret']:+.2%}")
        print(f"  Ann. Volatility : {metrics['ann_vol']:.2%}")
        print(f"  Sharpe Ratio    : {metrics['sharpe']:.2f}")
        print(f"  Max Drawdown    : {metrics['max_dd']:.2%}")
        print(f"  Calmar Ratio    : {metrics['calmar']:.2f}")
        print(f"  Win Rate        : {metrics['win_rate']:.2%}")
        print(f"  Avg Turnover    : {metrics['avg_turnover']:.2%}")
        print("-" * 60)
        print(f"  Benchmark Ann.  : {metrics['bench_ann']:+.2%}")
        print(f"  Excess Total    : {metrics['excess_total']:+.2%}")
        print("=" * 60)

    def plot_equity(self, daily_ret, bench_daily, oos_dates):
        """绘制策略 vs 基准净值曲线。"""
        equity = np.cumprod(1 + daily_ret)
        bench_equity = np.cumprod(1 + bench_daily)

        dates = [str(d) for d in oos_dates]
        x = range(len(dates))

        plt.style.use("bmh")
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(x, equity, label="Strategy (Open-to-Open)", linewidth=1.5)
        ax.plot(x, bench_equity, label="Benchmark (HS300 equal-wt)", alpha=0.5, linewidth=1)

        # X 轴日期标签（稀疏化）
        step = max(1, len(dates) // 10)
        tick_pos = list(range(0, len(dates), step))
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([dates[i] for i in tick_pos], rotation=30, fontsize=8)

        ann_ret = (equity[-1] ** (252 / len(equity)) - 1) if len(equity) > 0 else 0
        sharpe_val = 0
        if np.std(daily_ret) > 0:
            sharpe_val = (np.mean(daily_ret) * 252 - 0.02) / (np.std(daily_ret) * np.sqrt(252) + 1e-6)

        ax.set_title(f"OOS Backtest: Ann Ret {ann_ret:.1%} | Sharpe {sharpe_val:.2f}")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        # 保存
        report_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports"
        )
        os.makedirs(report_dir, exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(report_dir, f"oos_performance_{run_id}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Equity curve saved: {path}")

    @staticmethod
    def _rolling_mean_1d(x, window):
        """对 1D tensor 计算滚动均值。"""
        T = x.shape[0]
        if T < window:
            return torch.zeros_like(x)
        pad = torch.zeros(window - 1, device=x.device)
        x_pad = torch.cat([pad, x])
        windows = x_pad.unfold(0, window, 1)
        return windows.mean(dim=-1)
