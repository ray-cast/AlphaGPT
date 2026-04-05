"""样本外业绩评估模块：输出年化收益、夏普、最大回撤等指标，并绘制净值曲线。"""

import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch

from .config import ModelConfig
from .backtest import AshareBacktest


class StrategyReport:
    def __init__(self, loader):
        self.loader = loader
        self.top_n = ModelConfig.TOP_N_STOCKS
        self.buy_cost = ModelConfig.TOTAL_BUY_COST
        self.sell_cost = ModelConfig.TOTAL_SELL_COST
        self.min_turnover = ModelConfig.MIN_TURNOVER_RATE

    def evaluate(self, alpha_values, split_type="val"):
        """
        在指定区间评估策略表现。

        Args:
            alpha_values: [num_stocks, T] 来自 PrefixVM 的 alpha 分值
            split_type: "test"（测试集）或 "val"（验证集）
        Returns:
            metrics: dict，含 ann_ret / ann_vol / sharpe / max_dd / calmar / turnover
            daily_ret: np.ndarray [T_oos] 每日组合收益
            bench_ret: np.ndarray [T_oos] 每日基准收益
            oos_dates: list of date strings
        """
        loader = self.loader
        if split_type == "test":
            start = loader.train_idx
            end = loader.test_idx
        else:  # "val" = 验证集 (17-18)
            start = 0
            end = loader.valid_idx
        target_ret = loader.target_ret  # [num_stocks, T]
        turnover_rate = loader.raw_data_cache.get(
            "turnover_rate", torch.zeros_like(alpha_values)
        )
        suspended_all = loader.raw_data_cache.get(
            "suspended", torch.zeros_like(alpha_values, dtype=torch.bool)
        )
        constituent_all = loader.raw_data_cache.get(
            "constituent", torch.ones_like(alpha_values, dtype=torch.bool)
        )

        alpha_oos = alpha_values[:, start:end]
        target_oos = target_ret[:, start:end]
        turnover_oos = turnover_rate[:, start:end]
        suspended_oos = suspended_all[:, start:end]
        constituent_oos = constituent_all[:, start:end]
        oos_dates = loader.dates[start:end]

        N, T = alpha_oos.shape

        # 排除停牌/非成分股（NaN 信号、NaN 收益、停牌、或非时点成分股）
        valid_mask = ~(torch.isnan(alpha_oos) | torch.isnan(target_oos) | suspended_oos) & constituent_oos

        # 构建持仓（复用 AshareBacktest 的逻辑，含动态仓位管理）
        valid_target = target_oos.clone()
        valid_target[~valid_mask] = 0.0
        valid_count = valid_mask.float().sum(dim=0).clamp(min=1)
        market_daily_ret = valid_target.sum(dim=0) / valid_count
        market_ma20 = self._rolling_mean_1d(market_daily_ret, 20)

        position = torch.zeros_like(alpha_oos)
        # 与 AshareBacktest 一致：排除停牌+低换手后选股（含换手阈值）
        scores_filter = alpha_oos.clone()
        scores_filter[~valid_mask] = float('-inf')
        scores_filter[turnover_oos <= self.min_turnover] = float('-inf')
        rank_gap = ModelConfig.REBALANCE_RANK_GAP
        position = AshareBacktest._build_position(
            scores_filter, valid_mask, min(self.top_n, N), rank_gap
        )

        # 熊市减仓系数：市场弱势时收益按半仓计算（不影响实际换手）
        bear_mask = market_ma20 < 0
        bear_scale = torch.ones(T, device=alpha_oos.device)
        bear_scale[bear_mask] = 0.5

        # 换手（基于实际选股变化，不含熊市虚拟减仓）
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0.0
        turnover = torch.abs(position - prev_pos)
        avg_cost = (self.buy_cost + self.sell_cost) / 2.0
        tx_cost = turnover * avg_cost

        # 组合收益（熊市减仓应用到收益端，不产生虚假换手成本）
        gross_pnl = position * target_oos * bear_scale.unsqueeze(0)
        net_pnl = gross_pnl - tx_cost

        daily_ret = net_pnl.sum(dim=0).cpu().numpy() / (position.sum(dim=0).clamp(min=1).cpu().numpy())

        # 基准：优先使用真实 HS300 指数日收益率，退化为等权基准
        if self.loader.benchmark_ret is not None:
            bench_daily = self.loader.benchmark_ret[start:end].cpu().numpy()
        else:
            bench_daily = market_daily_ret.cpu().numpy()  # 等权组合（旧逻辑）

        # 统计
        metrics = self._compute_metrics(daily_ret, bench_daily, turnover, T, oos_dates)
        return metrics, daily_ret, bench_daily, oos_dates

    def _compute_metrics(self, daily_ret, bench_daily, turnover, T, oos_dates=None):
        equity = np.cumprod(1 + daily_ret)
        bench_equity = np.cumprod(1 + bench_daily)

        total_ret = equity[-1] - 1
        ann_ret = equity[-1] ** (252 / T) - 1 if T > 0 else 0.0
        ann_vol = np.std(daily_ret) * np.sqrt(252)
        sharpe = (ann_ret - 0.02) / (ann_vol + 1e-6)

        # 最大回撤及区间
        dd = 1 - equity / np.maximum.accumulate(equity)
        max_dd = np.max(dd)
        max_dd_end = int(np.argmax(dd))
        running_max = np.maximum.accumulate(equity)
        peak_before_trough = running_max[max_dd_end]
        # 回撤起点 = 该谷值之前最后一个净值为 peak_before_trough 的位置
        peak_indices = np.where(equity[:max_dd_end + 1] == peak_before_trough)[0]
        max_dd_start = int(peak_indices[-1]) if len(peak_indices) > 0 else 0
        calmar = ann_ret / (max_dd + 1e-6)

        # 分年最大回撤
        yearly_dd = {}
        if oos_dates is not None:
            years = np.array([str(d)[:4] for d in oos_dates])
            for yr in sorted(set(years)):
                mask = years == yr
                if mask.sum() == 0:
                    continue
                yr_eq = equity[mask]
                yr_peak = np.maximum.accumulate(yr_eq)
                yr_dd = 1 - yr_eq / yr_peak
                yearly_dd[yr] = float(np.max(yr_dd))

        # 基准统计
        bench_total = bench_equity[-1] - 1
        bench_ann = bench_equity[-1] ** (252 / T) - 1 if T > 0 else 0.0

        # 超额统计（用策略净值/基准净值，而非差值累乘）
        strategy_equity = np.cumprod(1 + daily_ret)
        benchmark_equity = np.cumprod(1 + bench_daily)
        excess_equity = strategy_equity / np.maximum(benchmark_equity, 1e-12)
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
            "max_dd_start": max_dd_start,
            "max_dd_end": max_dd_end,
            "yearly_dd": yearly_dd,
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
        dd_start = oos_dates[metrics['max_dd_start']] if metrics['max_dd_start'] < len(oos_dates) else "N/A"
        dd_end = oos_dates[metrics['max_dd_end']] if metrics['max_dd_end'] < len(oos_dates) else "N/A"
        print(f"  Max DD Period   : {dd_start} ~ {dd_end}")
        print(f"  Calmar Ratio    : {metrics['calmar']:.2f}")
        print(f"  Win Rate        : {metrics['win_rate']:.2%}")
        print(f"  Avg Turnover    : {metrics['avg_turnover']:.2%}")
        print("-" * 60)
        print(f"  Benchmark Ann.  : {metrics['bench_ann']:+.2%}")
        print(f"  Excess Total    : {metrics['excess_total']:+.2%}")
        # 分年回撤
        yearly_dd = metrics.get("yearly_dd", {})
        if yearly_dd:
            print("-" * 60)
            print("  Yearly Max Drawdown:")
            for yr, dd_val in yearly_dd.items():
                print(f"    {yr}  : {dd_val:.2%}")
        print("=" * 60)

    def plot_equity(self, daily_ret, bench_daily, oos_dates, suffix=""):
        """绘制策略 vs 基准净值曲线，含回撤子图。"""
        equity = np.cumprod(1 + daily_ret)
        bench_equity = np.cumprod(1 + bench_daily)

        # 回撤序列
        dd = 1 - equity / np.maximum.accumulate(equity)

        dates = [str(d) for d in oos_dates]
        x = np.arange(len(dates))

        plt.style.use("bmh")
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 8), height_ratios=[3, 1], sharex=True,
            gridspec_kw={"hspace": 0.08},
        )

        # --- 上图：净值曲线 ---
        ax1.plot(x, equity, label="Strategy (Open-to-Open)", linewidth=1.5)
        bench_label = "HS300 Index" if self.loader.benchmark_ret is not None else "HS300 equal-wt"
        ax1.plot(x, bench_equity, label=f"Benchmark ({bench_label})", alpha=0.5, linewidth=1)

        # 标注最大回撤区间
        dd_peak_idx = int(np.argmax(dd))
        running_max = np.maximum.accumulate(equity)
        peak_val = running_max[dd_peak_idx]
        peak_indices = np.where(equity[:dd_peak_idx + 1] == peak_val)[0]
        dd_start_idx = int(peak_indices[-1]) if len(peak_indices) > 0 else 0

        ax1.axvspan(dd_start_idx, dd_peak_idx, color="red", alpha=0.15, label="Max Drawdown")
        ax1.plot(dd_start_idx, equity[dd_start_idx], "rv", markersize=8)
        ax1.plot(dd_peak_idx, equity[dd_peak_idx], "rv", markersize=8)
        max_dd_pct = dd[dd_peak_idx]
        mid = (dd_start_idx + dd_peak_idx) // 2
        ax1.annotate(
            f"Max DD: {max_dd_pct:.1%}\n{dates[dd_start_idx]}~{dates[dd_peak_idx]}",
            xy=(mid, equity[mid]),
            xytext=(mid, equity[dd_start_idx] * 0.92),
            fontsize=8, color="red", ha="center",
            arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
        )

        ann_ret = (equity[-1] ** (252 / len(equity)) - 1) if len(equity) > 0 else 0
        sharpe_val = 0
        if np.std(daily_ret) > 0:
            sharpe_val = (np.mean(daily_ret) * 252 - 0.02) / (np.std(daily_ret) * np.sqrt(252) + 1e-6)

        ax1.set_title(f"OOS Backtest: Ann Ret {ann_ret:.1%} | Sharpe {sharpe_val:.2f} | Max DD {max_dd_pct:.1%}")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.grid(True)
        ax1.set_ylabel("Equity")

        # --- 下图：回撤 ---
        ax2.fill_between(x, 0, -dd * 100, color="steelblue", alpha=0.5)
        ax2.plot(x, -dd * 100, color="steelblue", linewidth=0.6)
        ax2.axhline(0, color="gray", linewidth=0.5)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_ylim(min(-dd.max() * 100 * 1.2, -1), 0)

        # X 轴日期标签
        step = max(1, len(dates) // 10)
        tick_pos = list(range(0, len(dates), step))
        ax2.set_xticks(tick_pos)
        ax2.set_xticklabels([dates[i] for i in tick_pos], rotation=30, fontsize=8)

        plt.tight_layout()

        # 保存
        report_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports"
        )
        os.makedirs(report_dir, exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(report_dir, f"oos_performance_{run_id}{suffix}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Equity curve saved: {path}")

    @staticmethod
    def _rolling_mean_1d(x, window):
        """对 1D tensor 计算滚动均值，前期不足 window 时用 expanding mean。"""
        T = x.shape[0]
        if T < window:
            cumsum = torch.cumsum(x, dim=0)
            arange = torch.arange(1, T + 1, device=x.device, dtype=x.dtype)
            return cumsum / arange
        cumsum = torch.cumsum(x, dim=0)
        arange = torch.arange(1, T + 1, device=x.device, dtype=x.dtype)
        expanding_mean = cumsum / arange
        rolling_sum = cumsum[window - 1:] - torch.cat(
            [torch.zeros(1, device=x.device), cumsum[:T - window]]
        )
        rolling_mean = rolling_sum / window
        return torch.cat([expanding_mean[:window - 1], rolling_mean])
