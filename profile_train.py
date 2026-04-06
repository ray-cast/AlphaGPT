"""Line profiler for training pipeline — 找出性能瓶颈。

用法:
    python profile_train.py          # 完整 profiling
    python profile_train.py --quick  # 快速模式（少步数）

会生成 profile_result.txt 包含各函数逐行耗时。
"""

import argparse
import sys
import os

# 确保 line_profiler 可用
try:
    from line_profiler import LineProfiler
except ImportError:
    print("Installing line_profiler...")
    os.system(f"{sys.executable} -m pip install line_profiler -q")
    from line_profiler import LineProfiler

import torch
from model_core.engine import AlphaEngine
from model_core.engine import ModelConfig
from model_core.vm import PrefixVM
from model_core.backtest import AshareBacktest
from model_core import ops


def run_profile(quick=False):
    """运行带 line_profiler 的训练流程。"""
    steps = 5 if quick else 20

    print(f"=== Line Profiling ({steps} steps) ===\n")

    # --- 实例化引擎 ---
    engine = AlphaEngine(use_lord_regularization=False)

    # --- 准备 profiler ---
    lp = LineProfiler()

    # 1) Engine.train 中的关键路径
    lp.add_function(engine.train)
    lp.add_function(engine._get_strict_mask)
    lp.add_function(engine._valid_prefix_len)

    # 2) VM 执行
    lp.add_function(engine.vm.execute)

    # 3) Backtest 评估（当前版本的完整调用链）
    lp.add_function(engine.bt.evaluate)
    lp.add_function(engine.bt._build_position)
    lp.add_function(engine.bt._vectorized_ic)
    lp.add_function(engine.bt._rolling_mean_1d)

    # 4) 算子函数（JIT 编译的无法 profile，只加纯 Python 函数）
    lp.add_function(ops._ts_delta)
    lp.add_function(ops._ts_decay_linear)
    lp.add_function(ops._ts_rank)
    lp.add_function(ops._ts_min)
    lp.add_function(ops._ts_max)
    lp.add_function(ops._ts_std)
    lp.add_function(ops._ts_corr)
    lp.add_function(ops._cs_rank)

    # 启动 profiling
    lp.enable_by_count()

    # 临时修改 TRAIN_STEPS 以控制 profiling 时间
    original_steps = ModelConfig.TRAIN_STEPS
    original_patience = ModelConfig.PATIENCE_LIMIT
    ModelConfig.TRAIN_STEPS = steps
    ModelConfig.PATIENCE_LIMIT = steps + 10  # 防止早停

    try:
        engine.train()
    finally:
        ModelConfig.TRAIN_STEPS = original_steps
        ModelConfig.PATIENCE_LIMIT = original_patience

    # --- 输出结果 ---
    print("\n" + "=" * 80)
    print("PROFILING RESULTS")
    print("=" * 80)

    # 写入文件
    output_file = "profile_result.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        lp.print_stats(stream=f, stripzeros=True)

    # 同时打印到控制台
    lp.print_stats(stripzeros=True)

    print(f"\n详细结果已保存至 {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="快速模式(5步)")
    args = parser.parse_args()
    run_profile(quick=args.quick)
