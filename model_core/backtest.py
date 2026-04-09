import torch
from .config import ModelConfig


class AshareBacktest:
    """A股截面选股回测引擎。

    每个交易日对所有股票按 alpha 分值排序，做多排名前 N 只。
    已内置 T+1 规则（通过 open-to-open target_ret 实现）、
    佣金+印花税、涨跌停过滤、换手率过滤。
    """

    def __init__(self):
        self.commission = ModelConfig.COMMISSION_RATE
        self.slippage = ModelConfig.SLIPPAGE_RATE
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
            daily_pnl, factors, target_ret, valid_mask, position, N_stocks,
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

    def _compute_score(self, daily_pnl, factors, target_ret, valid_mask, position, N_stocks):
        """计算综合适应度：归一化加权 Sortino + IC - 回撤/活跃度惩罚。

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

        # 回撤惩罚（5% 起罚，梯度友好）
        cum_curve = torch.cumsum(daily_pnl, dim=0)
        running_max = torch.cummax(cum_curve, dim=0)[0]
        drawdown = cum_curve - running_max
        max_dd = drawdown.min()
        dd_penalty = torch.min(
            torch.relu(-max_dd - 0.05) * 3.0,
            torch.tensor(2.0, device=daily_pnl.device)
        )

        # IC 奖励（截面预测能力，线性 + 硬截断）
        ic_norm = 0.0
        if N_stocks > 10:
            raw_ic = self._vectorized_ic_raw(factors, target_ret, valid_mask)
            ic_norm = torch.clamp(raw_ic * 5.0, -1.0, 1.0)

        # 加权合成：sortino 3x + IC 2.5x - 各惩罚项
        fitness = 3.0 * sortino_norm + 2.5 * ic_norm - dd_penalty - activity_penalty
        fitness = torch.clamp(fitness, -5.0, 5.0)
        return fitness, cum_ret.item()

    @staticmethod
    def _build_position(scores, valid_mask, top_n, rank_gap, rebalance_freq=20, limit_down=None):
        """构建持仓矩阵（纯 torch 实现，消除 GPU↔CPU 搬运）。

        非再平衡日：持仓沿用 + 剔除无效 + 向量化补缺（无内层循环）。
        再平衡日：rank_gap 贪心换仓保留顺序逻辑（仅 top_n × rebalance_days 次迭代）。

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
        N_stocks, T_len = scores.shape

        # ---- 阈值关闭：回退到原始 top-K 逻辑 ----
        if rank_gap <= 0:
            position = torch.zeros_like(scores)
            _, topk_idx = scores.topk(top_n, dim=0)
            position.scatter_(0, topk_idx, 1.0)
            position[~valid_mask] = 0.0
            return position

        # ---- 预计算（全 torch，不离开 GPU）----
        k = min(top_n * 2, N_stocks)
        _, topk_idx = scores.topk(k, dim=0)                    # [k, T]
        sorted_idx = scores.argsort(dim=0, descending=True)    # [N, T]
        ranks = torch.zeros(N_stocks, T_len, dtype=torch.long, device=scores.device)
        rank_vals = torch.arange(N_stocks, dtype=torch.long, device=scores.device)[:, None]
        ranks.scatter_(0, sorted_idx, rank_vals.expand_as(ranks))

        position = torch.zeros_like(scores)
        held = torch.zeros(N_stocks, dtype=torch.bool, device=scores.device)

        for t in range(T_len):
            # 移除无效持仓
            held &= valid_mask[:, t]

            is_rebalance = (rebalance_freq <= 1) or (t == 0) or (t % rebalance_freq == 0)
            held_count = int(held.sum())

            # 补充空缺席位（向量化，无内层 Python 循环）
            if held_count < top_n:
                cands = topk_idx[:, t]
                fill = cands[~held[cands]][:top_n - held_count]
                held[fill] = True
                held_count = top_n

            # 再平衡日：贪心换仓（最多 top_n 次 swap，顺序逻辑无法向量化）
            if is_rebalance and held_count == top_n:
                today_ranks = ranks[:, t]
                cand_list = topk_idx[:, t].tolist()

                held_idx = torch.where(held)[0]
                hr = today_ranks[held_idx].clone()
                if limit_down is not None:
                    hr[limit_down[held_idx, t]] = -1
                wi = int(hr.argmax())
                weakest = int(held_idx[wi])
                w_rank = int(today_ranks[weakest])

                for ci in cand_list:
                    if held[ci]:
                        continue
                    if w_rank - int(today_ranks[ci]) > rank_gap:
                        held[weakest] = False
                        held[ci] = True
                        # 重新找最弱（跌停保护）
                        held_idx = torch.where(held)[0]
                        hr = today_ranks[held_idx].clone()
                        if limit_down is not None:
                            hr[limit_down[held_idx, t]] = -1
                        wi = int(hr.argmax())
                        weakest = int(held_idx[wi])
                        w_rank = int(today_ranks[weakest])
                    else:
                        break

            position[held, t] = 1.0

        return position

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
