"""
A股每日策略运行器

用法:
    # 完整流程：更新数据 → 训练 → 输出信号
    python run_daily.py

    # 只更新数据，不训练
    python run_daily.py --update-only

    # 用已有公式直接生成信号（跳过训练）
    python run_daily.py --signal-only

    # 跳过数据更新，直接训练+信号
    python run_daily.py --skip-update
"""

import os
import sys
import json
import argparse
from datetime import datetime

import torch
from dotenv import load_dotenv

load_dotenv()


def update_data(token: str, data_dir: str):
    """增量更新 Tushare 数据。"""
    from data_download import TushareDownloader

    dl = TushareDownloader(token=token, data_dir=data_dir)

    print("[1/3] 更新沪深300成分股列表...")
    dl.fetch_constituents()

    print("[2/3] 增量更新日线数据...")
    today = datetime.now().strftime("%Y%m%d")
    dl.fetch_all(start_date="20150101", end_date=today)

    print("数据更新完成。")


def run_train():
    """运行因子挖掘训练 + 生成信号。"""
    from model_core.engine import AlphaEngine

    print("[3/3] 开始因子挖掘训练...")
    eng = AlphaEngine(use_lord_regularization=True)
    eng.train()
    eng.generate_signals()

    # OOS 业绩评估
    _run_report(eng)


def _run_report(eng):
    """运行 OOS 业绩评估并输出报告。"""
    from model_core.vm import PrefixVM
    from model_core.report import StrategyReport

    if eng.best_formula is None:
        print("未训练出有效公式，跳过业绩评估")
        return

    vm = PrefixVM()
    alpha_values = vm.execute(eng.best_formula, eng.loader.feat_tensor)
    if alpha_values is None:
        print("公式执行失败，跳过业绩评估")
        return

    report = StrategyReport(eng.loader)

    # 测试集评估
    metrics, daily_ret, bench_daily, oos_dates = report.evaluate(alpha_values, split_type="test")
    print("\n>>> 测试集评估")
    report.print_report(metrics, oos_dates)
    report.plot_equity(daily_ret, bench_daily, oos_dates, suffix="_test")

    # 验证集评估
    metrics, daily_ret, bench_daily, oos_dates = report.evaluate(alpha_values, split_type="val")
    print("\n>>> 验证集评估")
    report.print_report(metrics, oos_dates)
    report.plot_equity(daily_ret, bench_daily, oos_dates, suffix="_val")


def run_signal_only():
    """用已有训练结果直接生成信号。优先从 training_history.json 读取。"""
    from model_core.config import ModelConfig
    from model_core.data_loader import AshareDataLoader
    from model_core.vm import PrefixVM
    from model_core.signal_writer import SignalWriter
    from model_core.report import StrategyReport

    formula = None
    decoded = "N/A"
    score = "N/A"

    # 优先从 training_history.json 读取
    history_path = "training_history.json"
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
        formula = history.get("best_formula")
        decoded = history.get("best_decoded", "N/A")
        best_scores = history.get("best_score", [])
        score = best_scores[-1] if best_scores else "N/A"
        print(f"从 {history_path} 加载公式: {decoded}")
        print(f"  历史得分: {score}")

    # 回退到 best_ashare_strategy.json
    if formula is None:
        formula_path = "best_ashare_strategy.json"
        if os.path.exists(formula_path):
            with open(formula_path, "r") as f:
                info = json.load(f)
            formula = info["formula"]
            decoded = info.get("decoded", "N/A")
            score = info.get("score", "N/A")
            print(f"从 {formula_path} 加载公式: {decoded}")
            print(f"  历史得分: {score}")

    if formula is None:
        print("错误: 未找到训练结果，请先运行训练: python run_daily.py")
        sys.exit(1)

    loader = AshareDataLoader()
    loader.load_data()

    vm = PrefixVM()
    alpha_values = vm.execute(formula, loader.feat_tensor)

    if alpha_values is None:
        print("公式执行失败")
        sys.exit(1)

    writer = SignalWriter(loader)
    writer.write_signals(alpha_values, ModelConfig.SIGNAL_DIR)

    # OOS 业绩评估
    report = StrategyReport(loader)
    metrics, daily_ret, bench_daily, oos_dates = report.evaluate(alpha_values, split_type="test")
    print("\n>>> 测试集评估")
    report.print_report(metrics, oos_dates)
    report.plot_equity(daily_ret, bench_daily, oos_dates, suffix="_test")
    metrics, daily_ret, bench_daily, oos_dates = report.evaluate(alpha_values, split_type="val")
    print("\n>>> 验证集评估")
    report.print_report(metrics, oos_dates)
    report.plot_equity(daily_ret, bench_daily, oos_dates, suffix="_val")

    # 打印今日 Top30
    print_top_picks(loader, alpha_values)


def print_top_picks(loader, alpha_values):
    """打印最新交易日的 Top30 选股。"""
    if loader.test_idx is None or loader.test_idx >= alpha_values.shape[1]:
        print("无样本外数据")
        return

    last_idx = alpha_values.shape[1] - 1
    last_date = loader.dates[last_idx]
    scores = alpha_values[:, last_idx].cpu().numpy()

    ranked = sorted(
        zip(loader.stock_codes, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    print(f"\n{'='*55}")
    print(f"  最新交易日选股信号: {last_date}")
    print(f"{'='*55}")
    print(f"  {'排名':>4}  {'股票代码':<12}  {'信号分值':>10}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*10}")

    for rank, (code, score) in enumerate(ranked[:30], 1):
        direction = "做多" if score > 0 else "观望"
        print(f"  {rank:>4}  {code:<12}  {score:>+10.4f}  {direction}")

    print(f"  {'-'*4}  {'-'*12}  {'-'*10}")

    long_count = sum(1 for _, s in ranked[:30] if s > 0)
    print(f"  做多 {long_count}/30, 观望 {30 - long_count}/30")
    print(f"{'='*55}\n")


def main():
    parser = argparse.ArgumentParser(description="A股每日策略运行器")
    parser.add_argument("--update-only", action="store_true",
                        help="只更新数据，不训练")
    parser.add_argument("--signal-only", action="store_true",
                        help="用已有公式直接生成信号，跳过训练")
    parser.add_argument("--skip-update", action="store_true",
                        help="跳过数据更新，直接训练")
    args = parser.parse_args()

    token = os.getenv("TUSHARE_TOKEN", "")
    if not token and not args.skip_update and not args.signal_only:
        print("错误: .env 中未设置 TUSHARE_TOKEN")
        sys.exit(1)

    # ---- 数据更新 ----
    if not args.skip_update and not args.signal_only:
        update_data(token, "data")

    if args.update_only:
        print("\n数据更新完成，退出。")
        return

    # ---- 训练或生成信号 ----
    if args.signal_only:
        run_signal_only()
    else:
        run_train()


if __name__ == "__main__":
    main()
