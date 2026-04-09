import torch


# ---- JIT 编译的纯函数（从类方法提取） ----

@torch.jit.script
def robust_norm(t: torch.Tensor) -> torch.Tensor:
    """Expanding z-score 标准化，避免未来数据泄漏。NaN 位置保留 NaN。
    沿 dim=1（时间轴）使用 expanding window：时刻 i 的统计量只用 [0, i] 区间数据。
    用 cumsum 实现 O(T) 全向量化，替代原 O(T^2) nanmedian 循环。
    """
    nan_mask = torch.isnan(t)
    t_safe = torch.where(nan_mask, torch.zeros_like(t), t)
    valid_count = torch.cumsum((~nan_mask).float(), dim=1).clamp(min=1.0)

    cs = torch.cumsum(t_safe, dim=1)
    cs2 = torch.cumsum(t_safe * t_safe, dim=1)

    mean = cs / valid_count
    var = cs2 / valid_count - mean * mean
    std = torch.sqrt(var.clamp(min=1e-6))

    norm = (t_safe - mean) / std
    norm = torch.clamp(norm, -5.0, 5.0)
    norm = torch.where(nan_mask, torch.full_like(norm, float('nan')), norm)
    return norm


@torch.jit.script
def _rolling_mean(x: torch.Tensor, window: int) -> torch.Tensor:
    """沿 dim=1 的 expanding window 均值：前 window-1 天用 expanding，之后用固定窗口。
    NaN-safe：跳过 NaN 值，不传播到后续时间步。
    """
    N, T = x.shape
    nan_mask = torch.isnan(x)
    x_safe = torch.where(nan_mask, torch.zeros_like(x), x)
    valid_count = torch.cumsum((~nan_mask).float(), dim=1).clamp(min=1.0)

    cs = torch.cumsum(x_safe, dim=1)

    if T < window:
        result = cs / valid_count
        return torch.where(nan_mask, torch.full_like(result, float('nan')), result)

    # 前 window-1 天用 expanding mean
    expanding_mean = cs / valid_count
    # 第 window 天起用固定窗口滚动均值
    rolling_sum = cs[:, window - 1:] - torch.cat(
        [torch.zeros(N, 1, device=x.device), cs[:, :T - window]], dim=1)
    rolling_count = valid_count[:, window - 1:] - torch.cat(
        [torch.zeros(N, 1, device=x.device), valid_count[:, :T - window]], dim=1)
    rolling_count = rolling_count.clamp(min=1.0)
    rolling_mean = rolling_sum / rolling_count
    result = torch.cat([expanding_mean[:, :window - 1], rolling_mean], dim=1)
    # 原始 NaN 位置还原为 NaN
    result = torch.where(nan_mask, torch.full_like(result, float('nan')), result)
    return result


@torch.jit.script
def _rolling_sum(x: torch.Tensor, window: int) -> torch.Tensor:
    """沿 dim=1 的滚动求和，前期不足窗口时用累积和。"""
    N, T = x.shape
    if T < window:
        return torch.cumsum(x, dim=1)
    pad = torch.zeros(N, window - 1, device=x.device, dtype=x.dtype)
    x_pad = torch.cat([pad, x], dim=1)
    cs = torch.cumsum(x_pad, dim=1)
    rolling = cs[:, window - 1:] - torch.cat(
        [torch.zeros(N, 1, device=x.device, dtype=x.dtype), cs[:, :T - 1]], dim=1)
    expanding = torch.cumsum(x[:, :window - 1], dim=1)
    if T > window:
        return torch.cat([expanding, rolling[:, window - 1:]], dim=1)
    return rolling


@torch.jit.script
def _rolling_std(x: torch.Tensor, window: int) -> torch.Tensor:
    """沿 dim=1 的滚动标准差（cumsum 公式）。"""
    N, T = x.shape
    if T < window:
        return torch.zeros_like(x)
    x2 = x * x
    pad = torch.zeros(N, window - 1, device=x.device, dtype=x.dtype)
    x_pad = torch.cat([pad, x], dim=1)
    x2_pad = torch.cat([pad, x2], dim=1)
    cs = torch.cumsum(x_pad, dim=1)
    cs2 = torch.cumsum(x2_pad, dim=1)
    roll_sum = cs[:, window - 1:] - torch.cat(
        [torch.zeros(N, 1, device=x.device, dtype=x.dtype), cs[:, :T - 1]], dim=1)
    roll_sum2 = cs2[:, window - 1:] - torch.cat(
        [torch.zeros(N, 1, device=x.device, dtype=x.dtype), cs2[:, :T - 1]], dim=1)
    mean = roll_sum / window
    var = roll_sum2 / window - mean * mean
    std = torch.sqrt(var.clamp(min=1e-12))
    return torch.cat([torch.zeros(N, window - 1, device=x.device, dtype=x.dtype), std[:, window - 1:]], dim=1)


@torch.jit.script
def _rolling_max(x: torch.Tensor, window: int) -> torch.Tensor:
    """沿 dim=1 的滚动最大值（pad + unfold，输出长度 = T）。"""
    N, T = x.shape
    if T < window:
        return x
    pad = torch.full((N, window - 1), float('-inf'), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, window, 1)
    return windows.max(dim=-1)[0]


@torch.jit.script
def _rolling_min(x: torch.Tensor, window: int) -> torch.Tensor:
    """沿 dim=1 的滚动最小值（pad + unfold，输出长度 = T）。"""
    N, T = x.shape
    if T < window:
        return x
    pad = torch.full((N, window - 1), float('inf'), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, window, 1)
    return windows.min(dim=-1)[0]


@torch.jit.script
def _ema(x: torch.Tensor, span: int) -> torch.Tensor:
    """指数移动平均。alpha = 2 / (span + 1)，逐时间步递推。"""
    alpha = 2.0 / (span + 1)
    result = torch.zeros_like(x)
    result[:, 0] = x[:, 0]
    for t in range(1, x.shape[1]):
        result[:, t] = alpha * x[:, t] + (1.0 - alpha) * result[:, t - 1]
    return result


# ---- JIT 编译的技术指标函数 ----

@torch.jit.script
def _ind_liquidity_health(liquidity: torch.Tensor, fdv: torch.Tensor) -> torch.Tensor:
    ratio = liquidity / (fdv + 1e-6)
    return torch.clamp(ratio * 4.0, 0.0, 1.0)


@torch.jit.script
def _ind_buy_sell_imbalance(close: torch.Tensor, open_: torch.Tensor,
                            high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
    range_hl = high - low + 1e-9
    body = close - open_
    strength = body / range_hl
    return torch.tanh(strength * 3.0)


@torch.jit.script
def _ind_fomo_acceleration(volume: torch.Tensor) -> torch.Tensor:
    vol_prev = torch.roll(volume, 1, dims=1)
    vol_chg = (volume - vol_prev) / (vol_prev + 1.0)
    vol_chg[:, 0] = 0.0
    acc = vol_chg - torch.roll(vol_chg, 1, dims=1)
    acc[:, 0] = 0.0
    return torch.clamp(acc, -5.0, 5.0)


@torch.jit.script
def _ind_pump_deviation(close: torch.Tensor, window: int) -> torch.Tensor:
    ma = _rolling_mean(close, window)
    return (close - ma) / (ma + 1e-9)


@torch.jit.script
def _ind_volatility_clustering(close: torch.Tensor, window: int) -> torch.Tensor:
    ret = torch.log(close / (torch.roll(close, 1, dims=1) + 1e-9))
    ret[:, 0] = 0.0
    ret_sq = ret ** 2
    vol_ma = _rolling_mean(ret_sq, window)
    return torch.sqrt(vol_ma + 1e-9)


@torch.jit.script
def _ind_momentum_reversal(close: torch.Tensor, window: int) -> torch.Tensor:
    ret = torch.log(close / (torch.roll(close, 1, dims=1) + 1e-9))
    ret[:, 0] = 0.0
    mom = _rolling_mean(ret, window) * window
    mom_prev = torch.roll(mom, 1, dims=1)
    mom_prev[:, 0] = 0.0
    reversal = (mom * mom_prev < 0).float()
    return reversal


@torch.jit.script
def _ind_relative_strength(close: torch.Tensor, window: int) -> torch.Tensor:
    ret = close - torch.roll(close, 1, dims=1)
    ret[:, 0] = 0.0
    gains = torch.relu(ret)
    losses = torch.relu(-ret)
    avg_gain = _rolling_mean(gains, window)
    avg_loss = _rolling_mean(losses, window)
    rs = (avg_gain + 1e-9) / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return (rsi - 50) / 50


@torch.jit.script
def _ind_daily_return(close: torch.Tensor) -> torch.Tensor:
    ret = torch.log(close / (torch.roll(close, 1, dims=1) + 1e-9))
    ret[:, 0] = 0.0
    return ret


@torch.jit.script
def _ind_cumulative_return(close: torch.Tensor, window: int) -> torch.Tensor:
    ret_n = close / (torch.roll(close, window, dims=1) + 1e-9) - 1.0
    ret_n[:, :window] = 0.0
    return ret_n


@torch.jit.script
def _ind_volume_change(volume: torch.Tensor, window: int) -> torch.Tensor:
    vol_ma = _rolling_mean(volume, window)
    return volume / (vol_ma + 1e-9) - 1.0


@torch.jit.script
def _ind_amount_ratio(amount: torch.Tensor, window: int) -> torch.Tensor:
    amt_ma = _rolling_mean(amount, window)
    return amount / (amt_ma + 1e-9) - 1.0


@torch.jit.script
def _ind_trend(close: torch.Tensor, window: int) -> torch.Tensor:
    ma = _rolling_mean(close, window)
    return close / (ma + 1e-9) - 1.0


@torch.jit.script
def _ind_amount_acceleration(amount: torch.Tensor, window: int) -> torch.Tensor:
    amt_ratio = _ind_amount_ratio(amount, window)
    amt_chg_lag = torch.roll(amt_ratio, 1, dims=1)
    amt_chg_lag[:, 0] = 0.0
    return amt_ratio - amt_chg_lag


@torch.jit.script
def _ind_vol_cluster(close: torch.Tensor, short_window: int, long_window: int) -> torch.Tensor:
    ret = torch.log(close / (torch.roll(close, 1, dims=1) + 1e-9))
    ret[:, 0] = 0.0
    ret_sq = ret ** 2
    vol_short = _rolling_mean(ret_sq, short_window)
    vol_long = _rolling_mean(ret_sq, long_window)
    return vol_short / (vol_long + 1e-9)


@torch.jit.script
def _ind_hl_range(high: torch.Tensor, low: torch.Tensor, close: torch.Tensor) -> torch.Tensor:
    return (high - low) / (close + 1e-9)


@torch.jit.script
def _ind_close_position(close: torch.Tensor, high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
    return (close - low) / (high - low + 1e-9)


@torch.jit.script
def _ind_atr(high: torch.Tensor, low: torch.Tensor, close: torch.Tensor, window: int) -> torch.Tensor:
    prev_close = torch.roll(close, 1, dims=1)
    prev_close[:, 0] = close[:, 0]
    tr = torch.max(high - low, torch.max(
        (high - prev_close).abs(), (low - prev_close).abs()))
    return _ema(tr, window)


@torch.jit.script
def _ind_mfi(high: torch.Tensor, low: torch.Tensor, close: torch.Tensor,
             volume: torch.Tensor, window: int) -> torch.Tensor:
    tp = (high + low + close) / 3.0
    raw_mf = tp * volume
    prev_tp = torch.roll(tp, 1, dims=1)
    prev_tp[:, 0] = tp[:, 0]
    pos_mf = torch.where(tp > prev_tp, raw_mf, torch.zeros_like(raw_mf))
    neg_mf = torch.where(tp < prev_tp, raw_mf, torch.zeros_like(raw_mf))
    pos_sum = _rolling_sum(pos_mf, window)
    neg_sum = _rolling_sum(neg_mf, window)
    return 100.0 - 100.0 / (1.0 + pos_sum / (neg_sum + 1e-9))


@torch.jit.script
def _ind_macd(close: torch.Tensor, fast: int, slow: int) -> torch.Tensor:
    return _ema(close, fast) - _ema(close, slow)


@torch.jit.script
def _ind_bb_width(close: torch.Tensor, window: int) -> torch.Tensor:
    ma = _rolling_mean(close, window)
    std = _rolling_std(close, window)
    return 4.0 * std / (ma + 1e-9)


@torch.jit.script
def _ind_willr(high: torch.Tensor, low: torch.Tensor, close: torch.Tensor,
               window: int) -> torch.Tensor:
    hh = _rolling_max(high, window)
    ll = _rolling_min(low, window)
    return -(hh - close) / (hh - ll + 1e-9) * 100.0


@torch.jit.script
def _ind_obv(close: torch.Tensor, volume: torch.Tensor) -> torch.Tensor:
    direction = torch.sign(close - torch.roll(close, 1, dims=1))
    direction[:, 0] = 0.0
    return torch.cumsum(direction * volume, dim=1)


@torch.jit.script
def _ind_cmo(close: torch.Tensor, window: int) -> torch.Tensor:
    delta = close - torch.roll(close, 1, dims=1)
    delta[:, 0] = 0.0
    up = torch.where(delta > 0, delta, torch.zeros_like(delta))
    down = torch.where(delta < 0, -delta, torch.zeros_like(delta))
    up_sum = _rolling_sum(up, window)
    down_sum = _rolling_sum(down, window)
    return 100.0 * (up_sum - down_sum) / (up_sum + down_sum + 1e-9)


@torch.jit.script
def _ind_illiq(close: torch.Tensor, amount: torch.Tensor, window: int) -> torch.Tensor:
    ret = (close - torch.roll(close, 1, dims=1)) / (torch.roll(close, 1, dims=1) + 1e-9)
    ret[:, 0] = 0.0
    abs_ret = ret.abs()
    mean_abs_ret = _rolling_mean(abs_ret, window)
    mean_amount = _rolling_mean(amount, window)
    return mean_abs_ret / (mean_amount + 1e-9)


@torch.jit.script
def _ind_adv(amount: torch.Tensor, window: int) -> torch.Tensor:
    return _rolling_mean(amount, window)


class Indicators:
    @staticmethod
    def liquidity_health(liquidity, fdv):
        return _ind_liquidity_health(liquidity, fdv)

    @staticmethod
    def buy_sell_imbalance(close, open_, high, low):
        return _ind_buy_sell_imbalance(close, open_, high, low)

    @staticmethod
    def fomo_acceleration(volume, window=5):
        return _ind_fomo_acceleration(volume)

    @staticmethod
    def pump_deviation(close, window=20):
        return _ind_pump_deviation(close, window)

    @staticmethod
    def volatility_clustering(close, window=10):
        return _ind_volatility_clustering(close, window)

    @staticmethod
    def momentum_reversal(close, window=5):
        return _ind_momentum_reversal(close, window)

    @staticmethod
    def relative_strength(close, window=14):
        return _ind_relative_strength(close, window)

    @staticmethod
    def daily_return(close):
        return _ind_daily_return(close)

    @staticmethod
    def cumulative_return(close, window=5):
        return _ind_cumulative_return(close, window)

    @staticmethod
    def volume_change(volume, window=20):
        return _ind_volume_change(volume, window)

    @staticmethod
    def amount_ratio(amount, window=20):
        return _ind_amount_ratio(amount, window)

    @staticmethod
    def trend(close, window=60):
        return _ind_trend(close, window)

    @staticmethod
    def amount_acceleration(amount, window=20):
        return _ind_amount_acceleration(amount, window)

    @staticmethod
    def vol_cluster(close, short_window=5, long_window=20):
        return _ind_vol_cluster(close, short_window, long_window)

    @staticmethod
    def hl_range(high, low, close):
        return _ind_hl_range(high, low, close)

    @staticmethod
    def close_position(close, high, low):
        return _ind_close_position(close, high, low)

    @staticmethod
    def p_value(close, pe_ttm, roe):
        """价值锚 (EPS_TTM * ROE * 100)。pe_ttm/roe 为 None 时返回零张量。"""
        if pe_ttm is None or roe is None:
            return torch.zeros_like(close)
        eps_ttm = close / (pe_ttm + 1e-9)
        return eps_ttm * roe * 100.0

    @staticmethod
    def atr(high, low, close, window=14):
        return _ind_atr(high, low, close, window)

    @staticmethod
    def mfi(high, low, close, volume, window=14):
        return _ind_mfi(high, low, close, volume, window)

    @staticmethod
    def macd(close, fast=12, slow=26):
        return _ind_macd(close, fast, slow)

    @staticmethod
    def bb_width(close, window=20):
        return _ind_bb_width(close, window)

    @staticmethod
    def willr(high, low, close, window=14):
        return _ind_willr(high, low, close, window)

    @staticmethod
    def obv(close, volume):
        return _ind_obv(close, volume)

    @staticmethod
    def cmo(close, window=14):
        return _ind_cmo(close, window)

    @staticmethod
    def log_mcap(total_mv):
        """对数总市值（截面规模因子）。total_mv 单位：万元。"""
        if total_mv is None:
            return None
        return torch.log(total_mv.clamp(min=1.0))

    @staticmethod
    def illiq(close, amount, window=20):
        return _ind_illiq(close, amount, window)

    @staticmethod
    def adv(amount, window=20):
        return _ind_adv(amount, window)


class FeatureEngineer:
    FEATURES = [
        'RET', 'RET5', 'VOL_CHG', 'AMT_RATIO', 'TURN', 'PRESSURE', 'DEV',
        'RSI', 'TREND', 'HL_RANGE', 'CLOSE_POS', 'P_VALUE', 'VOL_CLUSTER',
        'ATR14', 'MFI14', 'MACD', 'BB_WIDTH', 'WILLR', 'OBV', 'CMO14',
        'LOG_MCAP', 'ILLIQ', 'ADV20',
    ]
    INPUT_DIM = len(FEATURES)

    @staticmethod
    def compute_features(raw_dict):
        """
        从原始 OHLCV 数据计算 20 维因子。

        输入 raw_dict 键: open, high, low, close, vol, amount, turnover_rate
          各自形状 [num_stocks, T]
        输出: [num_stocks, 20, T]
        """
        c = raw_dict["close"]
        o = raw_dict["open"]
        h = raw_dict["high"]
        l = raw_dict["low"]
        v = raw_dict["vol"]
        amt = raw_dict["amount"]
        turn = raw_dict["turnover_rate"]

        # ---- 原有 13 维因子 ----
        ret = Indicators.daily_return(c)
        ret5 = Indicators.cumulative_return(c, 5)
        vol_chg = Indicators.volume_change(v, 20)
        amt_ratio = Indicators.amount_ratio(amt, 20)
        turn_normed = robust_norm(turn)
        pressure = Indicators.buy_sell_imbalance(c, o, h, l)
        dev = Indicators.pump_deviation(c, 20)
        rel_strength = Indicators.relative_strength(c, 14)
        trend = Indicators.trend(c, 60)
        hl_range = Indicators.hl_range(h, l, c)
        close_pos = Indicators.close_position(c, h, l)
        p_value = Indicators.p_value(c, raw_dict.get("pe_ttm"), raw_dict.get("roe"))
        vol_cluster = Indicators.vol_cluster(c, 5, 20)

        # ---- 新增 7 维因子 ----
        atr14 = Indicators.atr(h, l, c, 14)
        mfi14 = Indicators.mfi(h, l, c, v, 14)
        macd_val = Indicators.macd(c, 12, 26)
        bb_width = Indicators.bb_width(c, 20)
        willr_val = Indicators.willr(h, l, c, 14)
        obv_val = Indicators.obv(c, v)
        cmo14 = Indicators.cmo(c, 14)

        # ---- 新增 3 维因子（规模/流动性） ----
        log_mcap = Indicators.log_mcap(raw_dict.get("total_mv"))
        illiq = Indicators.illiq(c, amt, 20)
        adv20 = Indicators.adv(amt, 20)

        # 对数市值：无数据时用零张量（截面排名无意义但不影响其他因子）
        if log_mcap is None:
            log_mcap = torch.zeros_like(c)

        features = torch.stack([
            robust_norm(ret),          # [0]  RET
            robust_norm(ret5),         # [1]  RET5
            robust_norm(vol_chg),      # [2]  VOL_CHG
            robust_norm(amt_ratio),    # [3]  AMT_RATIO
            turn_normed,               # [4]  TURN
            robust_norm(pressure),     # [5]  PRESSURE
            robust_norm(dev),          # [6]  DEV
            robust_norm(rel_strength), # [7]  RSI
            robust_norm(trend),        # [8]  TREND
            robust_norm(hl_range),     # [9]  HL_RANGE
            robust_norm(close_pos),    # [10] CLOSE_POS
            robust_norm(p_value),      # [11] P_VALUE
            robust_norm(vol_cluster),  # [12] VOL_CLUSTER
            robust_norm(atr14),        # [13] ATR14
            robust_norm(mfi14),        # [14] MFI14
            robust_norm(macd_val),     # [15] MACD
            robust_norm(bb_width),     # [16] BB_WIDTH
            robust_norm(willr_val),    # [17] WILLR
            robust_norm(obv_val),      # [18] OBV
            robust_norm(cmo14),        # [19] CMO14
            robust_norm(log_mcap),     # [20] LOG_MCAP
            robust_norm(illiq),        # [21] ILLIQ
            robust_norm(adv20),        # [22] ADV20
        ], dim=1)

        # 清理 Inf（但保留 NaN 标记停牌日）
        features = torch.nan_to_num(features, nan=float('nan'), posinf=5.0, neginf=-5.0)

        return features
