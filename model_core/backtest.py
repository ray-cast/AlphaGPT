import torch
from .config import ModelConfig
from .filter import apply_fundamental_filter


class AshareBacktest:
    """A股截面选股回测引擎。

    每个交易日对所有股票按 alpha 分值排序，做多排名前 N 只。
    已内置 T+1 规则（通过 open-to-open target_ret 实现）、
    佣金+印花税、涨跌停过滤、换手率过滤。
    """

    def __init__(self):
        self.buy_cost = ModelConfig.TOTAL_BUY_COST
        self.sell_cost = ModelConfig.TOTAL_SELL_COST
        self.min_turnover = ModelConfig.MIN_TURNOVER_RATE
        self.top_n = ModelConfig.TOP_N_STOCKS
        self.rebalance_freq = ModelConfig.REBALANCE_FREQ

    def evaluate(self, factors, raw_data, target_ret, start_idx=0, end_idx=None):
        """
        Args:
            factors:    [num_stocks, T] alpha 信号（来自 PrefixVM）
            raw_data:   dict，含 'turnover_rate', 'pct_chg'（如有）, 'vol', 'close'
            target_ret: [num_stocks, T] open-to-open 前向收益
            start_idx:  起始时间索引（含）
            end_idx:    结束时间索引（不含），None 表示到末尾
        Returns:
            (fitness: scalar tensor, avg_daily_return: float)
        """
        turnover_rate = raw_data.get("turnover_rate",
                            torch.zeros_like(factors))
        suspended = raw_data.get("suspended",
                            torch.zeros_like(factors, dtype=torch.bool))
        constituent = raw_data.get("constituent",
                            torch.ones_like(factors, dtype=torch.bool))
        pe_ttm_full = raw_data.get("pe_ttm")
        roe_full = raw_data.get("roe")

        # 按指定区间切片
        factors = factors[:, start_idx:end_idx]
        target_ret = target_ret[:, start_idx:end_idx]
        turnover_rate = turnover_rate[:, start_idx:end_idx]
        suspended = suspended[:, start_idx:end_idx]
        constituent = constituent[:, start_idx:end_idx]
        pe_ttm = pe_ttm_full[:, start_idx:end_idx] if pe_ttm_full is not None else None
        roe = roe_full[:, start_idx:end_idx] if roe_full is not None else None

        N_stocks, T_len = factors.shape

        # 排除停牌/非成分股（NaN 信号、NaN target_ret、停牌、或非时点成分股）
        valid_mask = ~(torch.isnan(factors) | torch.isnan(target_ret) | suspended) & constituent
        scores = factors.clone()
        scores[~valid_mask] = float('-inf')
        scores[turnover_rate <= self.min_turnover] = float('-inf')

        # 基本面过滤（训练用 soft penalty，保留探索空间）
        apply_fundamental_filter(scores, pe_ttm, roe, soft=True)

        # 市场状态判断：等权组合收益的 20 日均线（排除停牌）
        valid_target = target_ret.clone()
        valid_target[~valid_mask] = 0.0
        valid_count = valid_mask.float().sum(dim=0).clamp(min=1)
        market_daily_ret = valid_target.sum(dim=0) / valid_count
        market_ma20 = self._rolling_mean_1d(market_daily_ret, 20)

        # 构建持仓矩阵
        rank_gap = ModelConfig.REBALANCE_RANK_GAP
        position = self._build_position(scores, valid_mask, self.top_n, rank_gap, self.rebalance_freq)

        # 熊市缩仓：MA20<0 时统一缩减到 75%
        bear_mask = market_ma20 < 0
        stock_scale = torch.ones_like(position)
        if bear_mask.any():
            stock_scale[:, bear_mask] = 0.75

        # 换手
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0.0
        turnover = torch.abs(position - prev_pos)

        # 交易成本
        avg_cost = (self.buy_cost + self.sell_cost) / 2.0
        tx_cost = turnover * avg_cost

        # PnL：应用估值感知的缩减系数（不产生虚假换手成本）
        gross_pnl = position * target_ret * stock_scale
        net_pnl = gross_pnl - tx_cost

        # 每天 portfolio 收益（按实际持仓权重归一化，熊市减仓时分母会相应减小）
        daily_pnl = net_pnl.sum(dim=0) / (position.sum(dim=0).clamp(min=1))

        # ---- Fitness 计算：Sharpe + IC - 回撤惩罚 ----
        cum_ret = daily_pnl.sum()
        mean_ret = daily_pnl.mean()

        # 活跃度不足惩罚（硬性门槛，提前返回）
        active_days = (position.sum(dim=0) > 0).float().sum()
        if active_days < 10:
            return torch.tensor(-10.0, device=factors.device), 0.0

        # Sharpe（方向性 + 风险调整合一）
        daily_std = daily_pnl.std() + 1e-6
        sharpe = mean_ret / daily_std
        sharpe_score = torch.tanh(sharpe * 2.0)

        # 回撤惩罚（连续化，梯度友好）
        cum_curve = torch.cumsum(daily_pnl, dim=0)
        running_max = torch.cummax(cum_curve, dim=0)[0]
        drawdown = cum_curve - running_max
        max_dd = drawdown.min()
        dd_penalty = torch.relu(-max_dd - 0.05) * 2.0

        # IC 奖励（截面预测能力）
        ic_bonus = 0.0
        if N_stocks > 10:
            ic_bonus = self._vectorized_ic(factors, target_ret, valid_mask, T_len, N_stocks)

        # 综合得分：Sharpe 60% + IC 40% - 回撤惩罚
        fitness = 0.6 * sharpe_score + 0.4 * ic_bonus - dd_penalty

        return fitness, cum_ret.item()

    @staticmethod
    def _build_position(scores, valid_mask, top_n, rank_gap, rebalance_freq=20):
        """构建持仓矩阵，引入再平衡周期和换手惩罚阈值。

        每 rebalance_freq 个交易日执行一次截面选股（再平衡），
        非再平衡日沿用上一日持仓（仅剔除停牌/退成分股）。
        再平衡日内保留上一日持仓，只有当新候选股票的截面排名领先当前最弱持仓
        超过 rank_gap 名时才换仓，以减少无谓换手和交易成本。

        优化：使用 numpy boolean mask + 批量赋值，避免逐元素 GPU tensor 操作。

        Args:
            scores:        [N_stocks, T] 分数（无效股票已设 -inf）
            valid_mask:    [N_stocks, T] bool 有效股票掩码
            top_n:         持仓数量
            rank_gap:      换仓排名阈值（0 = 每日完全重选，即关闭阈值）
            rebalance_freq: 再平衡周期（交易日），1 = 每日再平衡
        Returns:
            position:   [N_stocks, T] 持仓矩阵（0/1）
        """
        import numpy as np

        N_stocks, T_len = scores.shape

        # ---- 阈值关闭：回退到原始 top-K 逻辑 ----
        if rank_gap <= 0:
            position = torch.zeros_like(scores)
            _, topk_idx = scores.topk(top_n, dim=0)
            position.scatter_(0, topk_idx, 1.0)
            position[~valid_mask] = 0.0
            return position

        # ---- 预计算：一次性转 numpy，避免循环中反复 GPU→CPU ----
        # 直接在 numpy 上做 argsort/topk，避免 GPU sync 开销
        scores_np = scores.cpu().numpy()            # [N_stocks, T_len]
        valid_np = valid_mask.cpu().numpy()         # [N_stocks, T_len]

        # numpy 排名表（0 = 最佳排名）
        sorted_idx_np = scores_np.argsort(axis=0)[::-1]  # descending
        ranks_np = np.zeros_like(sorted_idx_np, dtype=np.int64)
        rank_pos_np = np.arange(N_stocks, dtype=np.int64)[:, np.newaxis]
        ranks_np[sorted_idx_np, np.arange(T_len)] = rank_pos_np

        # numpy topk（按天取前 k 名，按分数降序排列）
        k = min(top_n * 2, N_stocks)
        topk_np = np.zeros((k, T_len), dtype=np.int64)
        for t_col in range(T_len):
            # argsort 降序取前 k 个（N=300 时 ~10μs，可接受）
            topk_np[:, t_col] = np.argsort(-scores_np[:, t_col])[:k]
        pos_np = np.zeros((N_stocks, T_len), dtype=np.float32)

        # ---- 用 boolean mask 跟踪持仓 ----
        held_mask = np.zeros(N_stocks, dtype=bool)

        for t in range(T_len):
            # 移除无效持仓：一条 numpy AND 操作
            held_mask &= valid_np[:, t]

            is_rebalance = (rebalance_freq <= 1) or (t == 0) or (t % rebalance_freq == 0)

            # 补充空缺席位（停牌导致脱落时）
            held_count = int(held_mask.sum())
            if held_count < top_n:
                candidates = topk_np[:, t]  # 当天预计算的 topk
                missing = top_n - held_count
                for idx in candidates:
                    if missing <= 0:
                        break
                    if not held_mask[idx]:
                        held_mask[idx] = True
                        missing -= 1

            # 再平衡日：尝试升级持仓
            if is_rebalance and held_mask.sum() == top_n:
                candidates = topk_np[:, t]
                today_ranks = ranks_np[:, t]

                held_indices = np.where(held_mask)[0]
                # 找当前最弱持仓（排名最大 = 最差）
                held_ranks = today_ranks[held_indices]
                weakest_pos = int(np.argmax(held_ranks))
                weakest_idx = held_indices[weakest_pos]
                weakest_rank = int(today_ranks[weakest_idx])

                for cand_idx in candidates:
                    if held_mask[cand_idx]:
                        continue
                    cand_rank = int(today_ranks[cand_idx])
                    if weakest_rank - cand_rank > rank_gap:
                        # 换仓
                        held_mask[weakest_idx] = False
                        held_mask[cand_idx] = True
                        # 重新找最弱
                        held_indices = np.where(held_mask)[0]
                        held_ranks = today_ranks[held_indices]
                        weakest_pos = int(np.argmax(held_ranks))
                        weakest_idx = held_indices[weakest_pos]
                        weakest_rank = int(today_ranks[weakest_idx])
                    else:
                        break

            # 批量赋值：一次 numpy 操作代替逐元素 GPU 写入
            pos_np[held_mask, t] = 1.0

        return torch.from_numpy(pos_np).to(device=scores.device, dtype=scores.dtype)

    @staticmethod
    def _vectorized_ic(factors, target_ret, valid_mask, T_len, N_stocks):
        """向量化计算逐日 Pearson 秩相关 IC（近似 Spearman）。

        直接用 Pearson 相关系数代替 Spearman，避免 argsort 开销。
        对于截面选股来说两者高度一致（同调单调变换不改变排序）。
        """
        # 每日有效股票数
        n_valid = valid_mask.float().sum(dim=0).clamp(min=1)  # [T_len]

        # 将 invalid 位置置零
        f_c = factors.clone()
        r_c = target_ret.clone()
        f_c[~valid_mask] = 0.0
        r_c[~valid_mask] = 0.0

        # 中心化（只对 valid 位置）
        f_mean = f_c.sum(dim=0) / n_valid
        r_mean = r_c.sum(dim=0) / n_valid
        f_c -= f_mean.unsqueeze(0)
        r_c -= r_mean.unsqueeze(0)
        f_c[~valid_mask] = 0.0
        r_c[~valid_mask] = 0.0

        # 逐日 Pearson 相关系数
        cov = (f_c * r_c).sum(dim=0)
        std_f = (f_c.pow(2).sum(dim=0)).sqrt().clamp(min=1e-6)
        std_r = (r_c.pow(2).sum(dim=0)).sqrt().clamp(min=1e-6)
        daily_ic = cov / (std_f * std_r)

        # 过滤无效天
        valid_day = (n_valid >= 10) & (std_f > 1e-3) & (std_r > 1e-3)
        ic_count = valid_day.float().sum()
        if ic_count < 1:
            return 0.0

        avg_ic = (daily_ic * valid_day.float()).sum() / ic_count
        return torch.tanh(avg_ic * 10.0).item()

    @staticmethod
    def _rolling_mean_1d(x, window):
        """对 1D tensor 计算滚动均值，前期不足 window 时用 expanding mean。"""
        T = x.shape[0]
        if T < window:
            # 不足一个窗口，全部用 expanding mean
            cumsum = torch.cumsum(x, dim=0)
            arange = torch.arange(1, T + 1, device=x.device, dtype=x.dtype)
            return cumsum / arange
        cumsum = torch.cumsum(x, dim=0)
        arange = torch.arange(1, T + 1, device=x.device, dtype=x.dtype)
        expanding_mean = cumsum / arange
        # 从第 window 个元素起用固定窗口
        rolling_sum = cumsum[window - 1:] - torch.cat(
            [torch.zeros(1, device=x.device), cumsum[:T - window]]
        )
        rolling_mean = rolling_sum / window
        return torch.cat([expanding_mean[:window - 1], rolling_mean])
