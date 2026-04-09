"""使用 line_profiler 分析 ops.py 和 factors.py 的逐行性能瓶颈。"""

import torch
import sys
import os

# 确保可以 import 项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_core.factors import (
    robust_norm, _rolling_mean, _rolling_sum, _rolling_std,
    _rolling_max, _rolling_min, _ema, Indicators, FeatureEngineer,
)
from model_core.ops import (
    _ts_delay, _signedpower, _op_gate, _op_jump, _op_decay,
    _ts_delta, _ts_decay_linear, _ts_rank, _ts_min, _ts_max,
    _ts_std, _ts_mean, _ts_corr, _ts_argmin, _ts_argmax, _ts_sum,
    _cs_rank, _cs_zscore, OPS_CONFIG,
)
from model_core.backtest import (
    AshareBacktest, _vectorized_ic_raw,
)
from model_core.config import ModelConfig

# ---------------------------------------------------------------------------
# line_profiler 不支持 JIT 函数，需创建纯 Python wrapper 来逐行计时
# ---------------------------------------------------------------------------

def _profile_ts_ops(x, y):
    """调用所有时序算子（纯 Python wrapper，可被 line_profiler 捕获）。"""
    d5, d10, d20, d60 = 5, 10, 20, 60
    r = {}
    r['ts_delay5']  = _ts_delay(x, d5)
    r['ts_delta5']  = _ts_delta(x, d5)
    r['ts_min5']    = _ts_min(x, d5)
    r['ts_max5']    = _ts_max(x, d5)
    r['ts_sum5']    = _ts_sum(x, d5)
    r['ts_argmin5'] = _ts_argmin(x, d5)
    r['ts_argmax5'] = _ts_argmax(x, d5)
    r['ts_mean5']   = _ts_mean(x, d5)

    r['ts_rank10']    = _ts_rank(x, d10)
    r['ts_decay10']   = _ts_decay_linear(x, d10)
    r['ts_delta10']   = _ts_delta(x, d10)

    r['ts_rank20']    = _ts_rank(x, d20)
    r['ts_decay20']   = _ts_decay_linear(x, d20)
    r['ts_std20']     = _ts_std(x, d20)
    r['ts_corr20']    = _ts_corr(x, y, d20)
    r['ts_mean20']    = _ts_mean(x, d20)

    r['ts_rank60']    = _ts_rank(x, d60)
    r['ts_decay60']   = _ts_decay_linear(x, d60)
    r['ts_std60']     = _ts_std(x, d60)
    r['ts_corr60']    = _ts_corr(x, y, d60)
    r['ts_mean60']    = _ts_mean(x, d60)
    r['ts_delta60']   = _ts_delta(x, d60)
    return r


def _profile_cs_ops(x):
    """调用所有截面算子。"""
    r = {}
    r['cs_rank']  = _cs_rank(x)
    r['cs_zscore'] = _cs_zscore(x)
    return r


def _profile_basic_ops(x, y):
    """调用基础算子。"""
    r = {}
    r['signedpower'] = _signedpower(x)
    r['op_gate']     = _op_gate(x, y, -y)
    r['op_jump']     = _op_jump(x)
    r['op_decay']    = _op_decay(x)
    return r


def _profile_rolling_helpers(close, volume, amount, turnover):
    """调用 factors.py 的滚动辅助函数。"""
    r = {}
    r['rolling_mean_20']  = _rolling_mean(close, 20)
    r['rolling_mean_60']  = _rolling_mean(close, 60)
    r['rolling_sum_14']   = _rolling_sum(volume, 14)
    r['rolling_std_20']   = _rolling_std(close, 20)
    r['rolling_max_14']   = _rolling_max(close, 14)
    r['rolling_min_14']   = _rolling_min(close, 14)
    r['ema_12']           = _ema(close, 12)
    r['ema_26']           = _ema(close, 26)
    r['robust_norm']      = robust_norm(close)
    return r


def _profile_indicators(close, open_, high, low, volume, amount, turnover):
    """调用所有 Indicators 静态方法。"""
    r = {}
    r['daily_return']       = Indicators.daily_return(close)
    r['cumulative_return']  = Indicators.cumulative_return(close, 5)
    r['volume_change']      = Indicators.volume_change(volume, 20)
    r['amount_ratio']       = Indicators.amount_ratio(amount, 20)
    r['buy_sell_imbalance'] = Indicators.buy_sell_imbalance(close, open_, high, low)
    r['pump_deviation']     = Indicators.pump_deviation(close, 20)
    r['relative_strength']  = Indicators.relative_strength(close, 14)
    r['trend']              = Indicators.trend(close, 60)
    r['hl_range']           = Indicators.hl_range(high, low, close)
    r['close_position']     = Indicators.close_position(close, high, low)
    r['vol_cluster']        = Indicators.vol_cluster(close, 5, 20)
    r['atr']                = Indicators.atr(high, low, close, 14)
    r['mfi']                = Indicators.mfi(high, low, close, volume, 14)
    r['macd']               = Indicators.macd(close, 12, 26)
    r['bb_width']           = Indicators.bb_width(close, 20)
    r['willr']              = Indicators.willr(high, low, close, 14)
    r['obv']                = Indicators.obv(close, volume)
    r['cmo']                = Indicators.cmo(close, 14)
    r['illiq']              = Indicators.illiq(close, amount, 20)
    r['adv']                = Indicators.adv(amount, 20)
    r['fomo_acceleration']  = Indicators.fomo_acceleration(volume, 5)
    r['momentum_reversal']  = Indicators.momentum_reversal(close, 5)
    return r


# ---------------------------------------------------------------------------
# 回测引擎性能测试（torch.profiler + 手动计时）
# ---------------------------------------------------------------------------

def _profile_backtest():
    """使用 torch.profiler + 手动计时分析回测引擎性能瓶颈。"""
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, T = 2000, 1500
    print(f"\n{'=' * 80}")
    print(f"回测引擎性能测试")
    print(f"设备: {device}，数据: {N} stocks × {T} days")
    print(f"{'=' * 80}")

    torch.manual_seed(42)

    # 合成回测数据
    factors = torch.randn(N, T, device=device)
    target_ret = (torch.randn(N, T, device=device) * 0.02).clamp(-0.1, 0.1)
    raw_data = {
        "turnover_rate":  (torch.rand(N, T, device=device) * 0.05 + 0.001).float(),
        "suspended":      torch.rand(N, T, device=device) > 0.95,
        "ipo_ok":         torch.rand(N, T, device=device) > 0.05,
    }

    bt = AshareBacktest()

    # Warmup
    print("Warmup JIT/CUDA...")
    bt.evaluate(factors.clone(), raw_data, target_ret.clone())
    if device.type == "cuda":
        torch.cuda.synchronize()

    def _timed(fn, *args, repeats=3, **kwargs):
        best = float('inf')
        result = None
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            best = min(best, time.perf_counter() - t0)
        return result, best

    # 准备 evaluate 内部所需的中间数据
    tr = raw_data["turnover_rate"]
    suspended = raw_data["suspended"]
    constituent = raw_data["ipo_ok"]
    valid_mask = ~(torch.isnan(factors) | torch.isnan(target_ret) | suspended) & constituent
    tradeable = valid_mask & (tr > bt.min_turnover)
    scores = torch.where(tradeable, factors, torch.tensor(float('-inf'), device=device))

    # 逐函数计时
    print("\n--- 逐函数计时 (3次取最快) ---")

    def _pnl():
        prev = torch.roll(position, 1, dims=1)
        prev[:, 0] = 0.0
        return ((position * target_ret - torch.abs(position - prev) * bt.commission)
                .sum(dim=0) / bt.top_n)

    position, t_pos = _timed(
        bt._build_position, scores, bt.top_n,
    )
    daily_pnl, t_pnl = _timed(_pnl)
    ic, t_ic = _timed(_vectorized_ic_raw, factors, target_ret, valid_mask)
    _, t_eval = _timed(bt.evaluate, factors.clone(), raw_data, target_ret.clone())

    results = [
        ("topk_position",      t_pos,  ""),
        ("pnl",                t_pnl,  ""),
        ("_vectorized_ic_raw", t_ic,   f"IC={ic.item():.4f}"),
        ("evaluate (总)",      t_eval, ""),
    ]
    name_w = max(len(n) for n, _, _ in results) + 2
    for name, t, extra in results:
        print(f"  {name:<{name_w}} {t:>8.4f}s  ({t / t_eval * 100:5.1f}%)  {extra}")

    # torch.profiler GPU kernel 级别分析
    print("\n--- torch.profiler kernel 分析 (3次 evaluate) ---")
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
    ) as prof:
        for _ in range(3):
            bt.evaluate(factors.clone(), raw_data, target_ret.clone())

    sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=15))


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    try:
        from line_profiler import LineProfiler
    except ImportError:
        print("line_profiler 未安装，正在安装...")
        os.system(f"{sys.executable} -m pip install line_profiler")
        from line_profiler import LineProfiler

    # 创建合成数据（模拟 2000 只股票 × 1500 个交易日）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, T = 2000, 1500
    print(f"设备: {device}，合成数据: {N} stocks × {T} days")

    torch.manual_seed(42)
    close   = (100 * torch.exp(torch.cumsum(torch.randn(N, T, device=device) * 0.02, dim=1))).float()
    open_   = close * (1 + torch.randn(N, T, device=device) * 0.005)
    high    = torch.max(close, open_) * (1 + torch.abs(torch.randn(N, T, device=device)) * 0.01)
    low     = torch.min(close, open_) * (1 - torch.abs(torch.randn(N, T, device=device)) * 0.01)
    volume  = (torch.rand(N, T, device=device) * 1e6 + 1e5).float()
    amount  = close * volume
    turnover = torch.rand(N, T, device=device).float()

    # 构造 raw_dict 用于 compute_features
    raw_dict = {
        "close": close, "open": open_, "high": high, "low": low,
        "vol": volume, "amount": amount, "turnover_rate": turnover,
        "pe_ttm": (torch.rand(N, T, device=device) * 50 + 5).float(),
        "roe": (torch.rand(N, T, device=device) * 0.3 - 0.05).float(),
        "total_mv": (torch.rand(N, T, device=device) * 1e7 + 1e5).float(),
    }

    # x, y 用于 ops 测试
    x = close.clone()
    y = volume.clone() / volume.max()

    # Warmup（JIT 编译 + CUDA 缓存）
    print("Warmup JIT/CUDA...")
    _ts_rank(x, 20)
    _ts_corr(x, y, 20)
    _cs_rank(x)
    _rolling_mean(close, 20)
    _ema(close, 12)
    robust_norm(close)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # ---- 构建 LineProfiler ----
    lp = LineProfiler()

    # 注册 ops wrapper
    lp.add_function(_profile_ts_ops)
    lp.add_function(_profile_cs_ops)
    lp.add_function(_profile_basic_ops)

    # 注册 factors 滚动辅助函数
    lp.add_function(robust_norm)
    lp.add_function(_rolling_mean)
    lp.add_function(_rolling_sum)
    lp.add_function(_rolling_std)
    lp.add_function(_rolling_max)
    lp.add_function(_rolling_min)
    lp.add_function(_ema)
    lp.add_function(_profile_rolling_helpers)

    # 注册 Indicators 方法
    lp.add_function(Indicators.daily_return)
    lp.add_function(Indicators.cumulative_return)
    lp.add_function(Indicators.volume_change)
    lp.add_function(Indicators.amount_ratio)
    lp.add_function(Indicators.buy_sell_imbalance)
    lp.add_function(Indicators.pump_deviation)
    lp.add_function(Indicators.relative_strength)
    lp.add_function(Indicators.trend)
    lp.add_function(Indicators.hl_range)
    lp.add_function(Indicators.close_position)
    lp.add_function(Indicators.vol_cluster)
    lp.add_function(Indicators.atr)
    lp.add_function(Indicators.mfi)
    lp.add_function(Indicators.macd)
    lp.add_function(Indicators.bb_width)
    lp.add_function(Indicators.willr)
    lp.add_function(Indicators.obv)
    lp.add_function(Indicators.cmo)
    lp.add_function(Indicators.illiq)
    lp.add_function(Indicators.adv)
    lp.add_function(Indicators.fomo_acceleration)
    lp.add_function(Indicators.momentum_reversal)
    lp.add_function(_profile_indicators)

    # 注册 compute_features
    lp.add_function(FeatureEngineer.compute_features)

    # 包装主调用入口
    wrapped_main = lp(_run_all)(
        x, y, close, open_, high, low, volume, amount, turnover, raw_dict
    )

    # 打印结果
    print("\n" + "=" * 80)
    print("LINE PROFILER 结果")
    print("=" * 80)
    lp.print_stats(stream=sys.stdout, stripzeros=True)

    # ---- 回测引擎性能测试 ----
    _profile_backtest()


def _run_all(x, y, close, open_, high, low, volume, amount, turnover, raw_dict):
    """所有被 profiling 的函数在此统一调用。"""
    _profile_basic_ops(x, y)
    _profile_ts_ops(x, y)
    _profile_cs_ops(x)
    _profile_rolling_helpers(close, volume, amount, turnover)
    _profile_indicators(close, open_, high, low, volume, amount, turnover)
    FeatureEngineer.compute_features(raw_dict)


if __name__ == "__main__":
    main()
