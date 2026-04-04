import torch
import torch.nn as nn


def robust_norm(t):
    """MAD 鲁棒标准化，截面维度（每只股票独立标准化）。NaN 位置保留 NaN。"""
    median = torch.nanmedian(t, dim=1, keepdim=True)[0]
    mad = torch.nanmedian(torch.abs(t - median), dim=1, keepdim=True)[0] + 1e-6
    norm = (t - median) / mad
    norm = torch.clamp(norm, -5.0, 5.0)
    # 保留原始 NaN 位置（停牌日）
    norm = torch.where(torch.isnan(t), torch.full_like(t, float('nan')), norm)
    return norm


def _rolling_mean(x, window):
    """沿 dim=1 的滚动均值。"""
    N, T = x.shape
    if T < window:
        return x
    pad = torch.zeros((N, window - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    return x_pad.unfold(1, window, 1).mean(dim=-1)


class FeatureEngineer:
    INPUT_DIM = 14

    @staticmethod
    def compute_features(raw_dict):
        """
        从原始 OHLCV 数据计算 14 维因子。

        输入 raw_dict 键: open, high, low, close, vol, amount, turnover_rate
          各自形状 [num_stocks, T]
        输出: [num_stocks, 14, T]
        """
        c = raw_dict["close"]
        o = raw_dict["open"]
        h = raw_dict["high"]
        l = raw_dict["low"]
        v = raw_dict["vol"]
        amt = raw_dict["amount"]
        turn = raw_dict["turnover_rate"]

        # ---- 因子 0: 日对数收益率 ----
        ret = torch.log(c / (torch.roll(c, 1, dims=1) + 1e-9))
        ret[:, 0] = 0.0

        # ---- 因子 1: 5日累计收益率 ----
        ret5 = c / (torch.roll(c, 5, dims=1) + 1e-9) - 1.0
        ret5[:, :5] = 0.0

        # ---- 因子 2: 成交量变化率 vs 20日均值 ----
        vol_ma20 = _rolling_mean(v, 20)
        vol_chg = v / (vol_ma20 + 1e-9) - 1.0

        # ---- 因子 3: 成交额比 vs 20日均值 ----
        amt_ma20 = _rolling_mean(amt, 20)
        amt_ratio = amt / (amt_ma20 + 1e-9) - 1.0

        # ---- 因子 4: 换手率 ----
        # 截面标准化：每天对所有股票排名
        turn_normed = robust_norm(turn)

        # ---- 因子 5: 买卖压力（K线实体/振幅） ----
        range_hl = h - l + 1e-9
        body = c - o
        pressure = torch.tanh((body / range_hl) * 3.0)

        # ---- 因子 6: 偏离 20日均线 ----
        ma20 = _rolling_mean(c, 20)
        dev = (c - ma20) / (ma20 + 1e-9)

        # ---- 因子 7: 相对强弱 RSI-like ----
        delta = c - torch.roll(c, 1, dims=1)
        delta[:, 0] = 0.0
        gains = torch.relu(delta)
        losses = torch.relu(-delta)
        window = 14
        avg_gain = _rolling_mean(gains, window)
        avg_loss = _rolling_mean(losses, window)
        rs = (avg_gain + 1e-9) / (avg_loss + 1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rel_strength = (rsi - 50.0) / 50.0  # 归一化到 [-1, 1]

        # ---- 因子 8: 价格 vs 60日均线（趋势） ----
        ma60 = _rolling_mean(c, 60)
        trend = c / (ma60 + 1e-9) - 1.0

        # ---- 因子 9: 成交额加速度（FOMO） ----
        amt_chg_lag = torch.roll(amt_ratio, 1, dims=1)
        amt_chg_lag[:, 0] = 0.0
        fomo = amt_ratio - amt_chg_lag

        # ---- 因子 10: 波动率聚集（VOL_CLUSTER） ----
        ret_sq = ret ** 2
        vol_short = _rolling_mean(ret_sq, 5)
        vol_long = _rolling_mean(ret_sq, 20)
        vol_cluster = vol_short / (vol_long + 1e-9)

        # ---- 因子 11: 高低价振幅（HL_RANGE） ----
        hl_range = (h - l) / (c + 1e-9)

        # ---- 因子 12: 收盘在区间位置（CLOSE_POS） ----
        close_pos = (c - l) / (h - l + 1e-9)

        # ---- 因子 13: 已实现波动率（REALIZED_VOL） ----
        realized_vol = torch.sqrt(vol_long + 1e-9)

        features = torch.stack([
            robust_norm(ret),          # [0] RET
            robust_norm(ret5),         # [1] RET5
            robust_norm(vol_chg),      # [2] VOL_CHG
            robust_norm(amt_ratio),    # [3] AMT_RAT
            turn_normed,               # [4] TURN
            pressure,                  # [5] PRESSURE
            robust_norm(dev),          # [6] DEV
            robust_norm(rel_strength), # [7] RSI
            robust_norm(trend),        # [8] TREND
            robust_norm(fomo),         # [9] FOMO
            robust_norm(vol_cluster),  # [10] VOL_CLUSTER
            robust_norm(hl_range),     # [11] HL_RANGE
            robust_norm(close_pos),    # [12] CLOSE_POS
            robust_norm(realized_vol), # [13] REALIZED_VOL
        ], dim=1)

        # 清理 Inf（但保留 NaN 标记停牌日）
        features = torch.nan_to_num(features, nan=float('nan'), posinf=5.0, neginf=-5.0)

        return features
