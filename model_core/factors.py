import torch


def robust_norm(t):
    """MAD 鲁棒标准化，expanding window 避免未来数据泄漏。NaN 位置保留 NaN。
    沿 dim=1（时间轴）使用 expanding window：时刻 i 的统计量只用 [0, i] 区间数据。
    """
    N, T = t.shape
    # 预计算 expanding median 和 expanding MAD（避免 look-ahead）
    median_out = torch.zeros_like(t)
    mad_out = torch.zeros_like(t)
    for i in range(T):
        col = t[:, :i + 1]  # 只用 [0, i]
        median_out[:, i] = torch.nanmedian(col, dim=1)[0]
        mad_out[:, i] = torch.nanmedian(torch.abs(col - median_out[:, i:i+1]), dim=1)[0] + 1e-6
    norm = (t - median_out) / mad_out
    norm = torch.clamp(norm, -5.0, 5.0)
    # 保留原始 NaN 位置（停牌日）
    norm = torch.where(torch.isnan(t), torch.full_like(t, float('nan')), norm)
    return norm


def _rolling_mean(x, window):
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

class Indicators:
    @staticmethod
    def liquidity_health(liquidity, fdv):
        ratio = liquidity / (fdv + 1e-6)
        return torch.clamp(ratio * 4.0, 0.0, 1.0)

    @staticmethod
    def buy_sell_imbalance(close, open_, high, low):
        range_hl = high - low + 1e-9
        body = close - open_
        strength = body / range_hl
        return torch.tanh(strength * 3.0)

    @staticmethod
    def fomo_acceleration(volume, window=5):
        vol_prev = torch.roll(volume, 1, dims=1)
        vol_chg = (volume - vol_prev) / (vol_prev + 1.0)
        vol_chg[:, 0] = 0.0
        acc = vol_chg - torch.roll(vol_chg, 1, dims=1)
        acc[:, 0] = 0.0
        return torch.clamp(acc, -5.0, 5.0)

    @staticmethod
    def pump_deviation(close, window=20):
        """偏离 window日均线"""
        ma = _rolling_mean(close, window)
        return (close - ma) / (ma + 1e-9)

    @staticmethod
    def volatility_clustering(close, window=10):
        """已实现波动率：滚动窗口 ret² 均值的平方根"""
        ret = torch.log(close / (torch.roll(close, 1, dims=1) + 1e-9))
        ret[:, 0] = 0.0
        ret_sq = ret ** 2
        vol_ma = _rolling_mean(ret_sq, window)
        return torch.sqrt(vol_ma + 1e-9)

    @staticmethod
    def momentum_reversal(close, window=5):
        """动量反转信号：当前动量与前一日符号相反"""
        ret = torch.log(close / (torch.roll(close, 1, dims=1) + 1e-9))
        ret[:, 0] = 0.0
        mom = _rolling_mean(ret, window) * window
        mom_prev = torch.roll(mom, 1, dims=1)
        mom_prev[:, 0] = 0.0
        reversal = (mom * mom_prev < 0).float()
        return reversal

    @staticmethod
    def relative_strength(close, window=14):
        """RSI-like 相对强弱指标"""
        ret = close - torch.roll(close, 1, dims=1)
        ret[:, 0] = 0.0
        gains = torch.relu(ret)
        losses = torch.relu(-ret)
        avg_gain = _rolling_mean(gains, window)
        avg_loss = _rolling_mean(losses, window)
        rs = (avg_gain + 1e-9) / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return (rsi - 50) / 50  # Normalize

    @staticmethod
    def daily_return(close):
        """日对数收益率"""
        ret = torch.log(close / (torch.roll(close, 1, dims=1) + 1e-9))
        ret[:, 0] = 0.0
        return ret

    @staticmethod
    def cumulative_return(close, window=5):
        """window日累计收益率"""
        ret_n = close / (torch.roll(close, window, dims=1) + 1e-9) - 1.0
        ret_n[:, :window] = 0.0
        return ret_n

    @staticmethod
    def volume_change(volume, window=20):
        """成交量变化率 vs window日均值"""
        vol_ma = _rolling_mean(volume, window)
        return volume / (vol_ma + 1e-9) - 1.0

    @staticmethod
    def amount_ratio(amount, window=20):
        """成交额比 vs window日均值"""
        amt_ma = _rolling_mean(amount, window)
        return amount / (amt_ma + 1e-9) - 1.0

    @staticmethod
    def trend(close, window=60):
        """价格 vs window日均线"""
        ma = _rolling_mean(close, window)
        return close / (ma + 1e-9) - 1.0

    @staticmethod
    def amount_acceleration(amount, window=20):
        """成交额加速度（FOMO）：成交额比的一阶差分"""
        amt_ratio = Indicators.amount_ratio(amount, window)
        amt_chg_lag = torch.roll(amt_ratio, 1, dims=1)
        amt_chg_lag[:, 0] = 0.0
        return amt_ratio - amt_chg_lag

    @staticmethod
    def vol_cluster(close, short_window=5, long_window=20):
        """波动率聚集（短期/长期波动率比）"""
        ret = torch.log(close / (torch.roll(close, 1, dims=1) + 1e-9))
        ret[:, 0] = 0.0
        ret_sq = ret ** 2
        vol_short = _rolling_mean(ret_sq, short_window)
        vol_long = _rolling_mean(ret_sq, long_window)
        return vol_short / (vol_long + 1e-9)

    @staticmethod
    def hl_range(high, low, close):
        """高低价振幅"""
        return (high - low) / (close + 1e-9)

    @staticmethod
    def close_position(close, high, low):
        """收盘在区间位置"""
        return (close - low) / (high - low + 1e-9)

    @staticmethod
    def p_value(close, pe_ttm, roe):
        """价值锚 (EPS_TTM × ROE × 100)"""
        if pe_ttm is None or roe is None:
            return torch.zeros_like(close)
        eps_ttm = close / (pe_ttm + 1e-9)
        return eps_ttm * roe * 100.0

class FeatureEngineer:
    INPUT_DIM = 13

    @staticmethod
    def compute_features(raw_dict):
        """
        从原始 OHLCV 数据计算 13 维因子。

        输入 raw_dict 键: open, high, low, close, vol, amount, turnover_rate
          各自形状 [num_stocks, T]
        输出: [num_stocks, 13, T]
        """
        c = raw_dict["close"]
        o = raw_dict["open"]
        h = raw_dict["high"]
        l = raw_dict["low"]
        v = raw_dict["vol"]
        amt = raw_dict["amount"]
        turn = raw_dict["turnover_rate"]

        # ---- 因子 0: 日对数收益率 ----
        ret = Indicators.daily_return(c)

        # ---- 因子 1: 5日累计收益率 ----
        ret5 = Indicators.cumulative_return(c, 5)

        # ---- 因子 2: 成交量变化率 vs 20日均值 ----
        vol_chg = Indicators.volume_change(v, 20)

        # ---- 因子 3: 成交额比 vs 20日均值 ----
        amt_ratio = Indicators.amount_ratio(amt, 20)

        # ---- 因子 4: 换手率 ----
        turn_normed = robust_norm(turn)

        # ---- 因子 5: 买卖压力（K线实体/振幅） ----
        pressure = Indicators.buy_sell_imbalance(c, o, h, l)

        # ---- 因子 6: 偏离 20日均线 ----
        dev = Indicators.pump_deviation(c, 20)

        # ---- 因子 7: 相对强弱 RSI-like ----
        rel_strength = Indicators.relative_strength(c, 14)

        # ---- 因子 8: 价格 vs 60日均线（趋势） ----
        trend = Indicators.trend(c, 60)

        # ---- 因子 9: 高低价振幅（HL_RANGE） ----
        hl_range = Indicators.hl_range(h, l, c)

        # ---- 因子 10: 收盘在区间位置（CLOSE_POS） ----
        close_pos = Indicators.close_position(c, h, l)

        # ---- 因子 11: 价值锚 P_value (EPS_TTM × ROE × 100) ----
        p_value = Indicators.p_value(c, raw_dict.get("pe_ttm"), raw_dict.get("roe"))

        # ---- 因子 12: 波动率聚集（短/长期波动率比） ----
        vol_cluster = Indicators.vol_cluster(c, 5, 20)

        features = torch.stack([
            robust_norm(ret),          # [0] RET
            robust_norm(ret5),         # [1] RET5
            robust_norm(vol_chg),      # [2] VOL_CHG
            robust_norm(amt_ratio),    # [3] AMT_RATIO
            turn_normed,               # [4] TURN
            robust_norm(pressure),     # [5] PRESSURE
            robust_norm(dev),          # [6] DEV
            robust_norm(rel_strength), # [7] RSI
            robust_norm(trend),        # [8] TREND
            robust_norm(hl_range),     # [9] HL_RANGE
            robust_norm(close_pos),    # [10] CLOSE_POS
            robust_norm(p_value),      # [11] P_VALUE
            robust_norm(vol_cluster),  # [12] VOL_CLUSTER
        ], dim=1)

        # 清理 Inf（但保留 NaN 标记停牌日）
        features = torch.nan_to_num(features, nan=float('nan'), posinf=5.0, neginf=-5.0)

        return features
