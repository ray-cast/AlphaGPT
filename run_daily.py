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


def run_signal_only():
    """用已有的 best_ashare_strategy.json 直接生成信号。"""
    from model_core.data_loader import AshareDataLoader
    from model_core.vm import StackVM
    from model_core.signal_writer import SignalWriter

    formula_path = "best_ashare_strategy.json"
    if not os.path.exists(formula_path):
        print(f"错误: 未找到 {formula_path}，请先运行训练: python run_daily.py")
        sys.exit(1)

    with open(formula_path, "r") as f:
        info = json.load(f)

    formula = info["formula"]
    print(f"加载已有公式: {info.get('decoded', 'N/A')}")
    print(f"  历史得分: {info.get('score', 'N/A')}")

    loader = AshareDataLoader()
    loader.load_data()

    vm = StackVM()
    alpha_values = vm.execute(formula, loader.feat_tensor)

    if alpha_values is None:
        print("公式执行失败")
        sys.exit(1)

    writer = SignalWriter(loader)
    writer.write_signals(alpha_values)

    # 打印今日 Top30
    print_top_picks(loader, alpha_values)


def print_top_picks(loader, alpha_values):
    """打印最新交易日的 Top30 选股。"""
    oos_start = loader.split_idx
    if oos_start is None or oos_start >= alpha_values.shape[1]:
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
