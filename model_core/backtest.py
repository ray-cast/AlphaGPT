import torch
from .config import ModelConfig


class AshareBacktest:
    """A股截面选股回测引擎。

    每个交易日对所有股票按 alpha 分值排序，做多排名前 N 只。
    已内置 T+1 规则（通过 open-to-open target_ret 实现）、
    佣金+印花税、涨跌停过滤、换手率过滤。
    """

    def __init__(self):
        self.commission = 0.00025
        self.slippage = 0.001
        self.buy_cost = self.commission + self.slippage   # 买入：佣金 + 滑点
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
        constituent = raw_data.get("ipo_ok",
                            torch.ones_like(factors, dtype=torch.bool))
        limit_up_full = raw_data.get("limit_up")
        limit_down_full = raw_data.get("limit_down")

        # 按指定区间切片
        factors = factors[:, start_idx:end_idx]
        target_ret = target_ret[:, start_idx:end_idx]
        turnover_rate = turnover_rate[:, start_idx:end_idx]
        suspended = suspended[:, start_idx:end_idx]
        constituent = constituent[:, start_idx:end_idx]
        limit_up = limit_up_full[:, start_idx:end_idx] if limit_up_full is not None else None
        limit_down = limit_down_full[:, start_idx:end_idx] if limit_down_full is not None else None

        N_stocks, T_len = factors.shape

        # 印花税按日期分段（2023-08-28 前千1，之后千0.5）
        stamp_tax_rate_full = raw_data.get("stamp_tax_rate")
        if stamp_tax_rate_full is not None:
            stamp_tax_rate = stamp_tax_rate_full[start_idx:end_idx]
        else:
            stamp_tax_rate = torch.full((T_len,), 0.0005, device=factors.device)

        # 排除停牌/非成分股（NaN 信号、NaN target_ret、停牌、或非时点成分股）
        valid_mask = ~(torch.isnan(factors) | torch.isnan(target_ret) | suspended) & constituent
        tradeable = valid_mask & (turnover_rate > self.min_turnover)
        scores = torch.where(tradeable, factors, torch.tensor(float('-inf'), device=factors.device))

        # 涨停过滤：涨停股票无法买入，从候选池排除
        if limit_up is not None:
            scores[limit_up] = float('-inf')

        # 构建持仓矩阵
        rank_gap = ModelConfig.REBALANCE_RANK_GAP
        position = self._build_position(scores, valid_mask, self.top_n, rank_gap, self.rebalance_freq,
                                        limit_down=limit_down)

        # 换手
        turnover = self._compute_turnover(position, valid_mask)

        # PnL（印花税按日期分段）
        daily_pnl = self._compute_pnl(position, target_ret, stamp_tax_rate)

        # 评分
        fitness, cum_ret = self._compute_score(
            daily_pnl, factors, target_ret, valid_mask, position, N_stocks, turnover,
        )
        return fitness, cum_ret, daily_pnl, turnover

    def _compute_turnover(self, position, valid_mask):
        """计算换手率矩阵。停牌/非成分股的仓位变动不计换手。"""
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0.0
        return torch.abs(position - prev_pos) * valid_mask.float()

    def _compute_pnl(self, position, target_ret, stamp_tax_rate):
        """计算每日组合收益（含交易成本）。买卖费率分离，印花税按日期分段。

        Args:
            stamp_tax_rate: [T] 每日印花税率（2023-08-28前千1，之后千0.5）
        """
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0.0
        buy_turnover = torch.clamp(position - prev_pos, min=0.0)
        sell_turnover = torch.clamp(prev_pos - position, min=0.0)
        sell_cost_t = (self.commission + stamp_tax_rate + self.slippage).unsqueeze(0)  # [1, T]
        tx_cost = buy_turnover * self.buy_cost + sell_turnover * sell_cost_t
        gross_pnl = position * target_ret
        net_pnl = gross_pnl - tx_cost
        return net_pnl.sum(dim=0) / self.top_n

    def _compute_score(self, daily_pnl, factors, target_ret, valid_mask, position, N_stocks, turnover):
        """计算综合适应度：归一化加权 Sortino + IC - 回撤/活跃度/换手惩罚。

        Args:
            turnover: 已在外部计算好的换手率矩阵，避免重复计算。

        Returns:
            (fitness: tensor, cum_ret: float)
        """
        cum_ret = daily_pnl.sum()
        mean_ret = daily_pnl.mean()

        # 活跃度不足惩罚（软惩罚，梯度友好）
        active_days = (position.sum(dim=0) > 0).float().sum()
        activity_penalty = torch.relu(10.0 - active_days) * 0.1

        # Sortino（只惩罚下行风险，不惩罚上行波动）
        downside = daily_pnl[daily_pnl < 0]
        if downside.numel() > 5:
            down_std = downside.std() + 1e-6
        else:
            down_std = daily_pnl.std() + 1e-6
        sortino = mean_ret / down_std * (252 ** 0.5)
        sortino_norm = torch.clamp(sortino, -3.0, 5.0) / 5.0  # [-0.6, 1.0]

        # 回撤惩罚（2% 起罚，梯度友好）
        cum_curve = torch.cumsum(daily_pnl, dim=0)
        running_max = torch.cummax(cum_curve, dim=0)[0]
        drawdown = cum_curve - running_max
        max_dd = drawdown.min()
        dd_penalty = torch.min(
            torch.relu(-max_dd - 0.02) * 3.0,
            torch.tensor(2.0, device=daily_pnl.device)
        )

        # IC 奖励（截面预测能力，线性 + 硬截断）
        ic_norm = 0.0
        if N_stocks > 10:
            raw_ic = self._vectorized_ic_raw(factors, target_ret, valid_mask)
            ic_norm = torch.clamp(raw_ic * 5.0, -1.0, 1.0)

        # 换手率惩罚（日均换手超 30% 起罚，封顶 1.0）
        avg_turnover = turnover.sum() / (active_days + 1e-6) / self.top_n
        turnover_penalty = torch.min(
            torch.relu(avg_turnover - 0.3) * 1.0,
            torch.tensor(1.0, device=daily_pnl.device)
        )

        # 加权合成：sortino 3x + IC 1.5x - 各惩罚项
        fitness = 3.0 * sortino_norm + 1.5 * ic_norm - dd_penalty - activity_penalty - turnover_penalty
        fitness = torch.clamp(fitness, -5.0, 5.0)
        return fitness, cum_ret.item()

    @staticmethod
    def _build_position(scores, valid_mask, top_n, rank_gap, rebalance_freq=20, limit_down=None):
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
            limit_down:    [N_stocks, T] bool 跌停掩码，跌停持仓不可被替换（卖出）
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

        # 跌停掩码转 numpy
        limit_down_np = limit_down.cpu().numpy() if limit_down is not None else None

        # 直接复用已排好的排名表取前 k 名（sorted_idx_np 降序，-inf 排末尾）
        k = min(top_n * 2, N_stocks)
        topk_np = sorted_idx_np[:k, :]
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
                # 找当前最弱持仓（排名最大 = 最差），跌停持仓不可被替换
                held_ranks = today_ranks[held_indices]
                if limit_down_np is not None:
                    held_blocked = limit_down_np[:, t][held_indices]
                    held_ranks = held_ranks.copy()
                    held_ranks[held_blocked] = -1  # 屏蔽：确保不被 argmax 选中
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
                        # 重新找最弱（跌停保护）
                        held_indices = np.where(held_mask)[0]
                        held_ranks = today_ranks[held_indices]
                        if limit_down_np is not None:
                            held_blocked = limit_down_np[:, t][held_indices]
                            held_ranks = held_ranks.copy()
                            held_ranks[held_blocked] = -1
                        weakest_pos = int(np.argmax(held_ranks))
                        weakest_idx = held_indices[weakest_pos]
                        weakest_rank = int(today_ranks[weakest_idx])
                    else:
                        break

            # 批量赋值：一次 numpy 操作代替逐元素 GPU 写入
            pos_np[held_mask, t] = 1.0

        return torch.from_numpy(pos_np).to(device=scores.device, dtype=scores.dtype)

    @staticmethod
    def _vectorized_ic_raw(factors, target_ret, valid_mask):
        """向量化计算逐日 Pearson IC，返回原始均值（由调用方决定映射方式）。"""
        n_valid = valid_mask.float().sum(dim=0).clamp(min=1)

        f_c = torch.where(valid_mask, factors, 0.0)
        r_c = torch.where(valid_mask, target_ret, 0.0)
        f_mean = f_c.sum(dim=0) / n_valid
        r_mean = r_c.sum(dim=0) / n_valid
        f_c = torch.where(valid_mask, f_c - f_mean.unsqueeze(0), 0.0)
        r_c = torch.where(valid_mask, r_c - r_mean.unsqueeze(0), 0.0)

        cov = (f_c * r_c).sum(dim=0)
        std_f = (f_c.pow(2).sum(dim=0)).sqrt().clamp(min=1e-6)
        std_r = (r_c.pow(2).sum(dim=0)).sqrt().clamp(min=1e-6)
        daily_ic = cov / (std_f * std_r)

        valid_day = (n_valid >= 10) & (std_f > 1e-3) & (std_r > 1e-3)
        ic_count = valid_day.float().sum()
        if ic_count < 1:
            return torch.tensor(0.0, device=factors.device)

        avg_ic = (daily_ic * valid_day.float()).sum() / ic_count
        return avg_ic

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
