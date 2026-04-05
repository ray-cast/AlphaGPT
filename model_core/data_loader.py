import os
import bisect
import glob
import pandas as pd
import torch
from .config import ModelConfig
from .factors import FeatureEngineer


class AshareDataLoader:
    """从 CSV 文件加载沪深300成分股日线数据，转为 PyTorch tensors。"""

    def __init__(self, data_dir: str = None, max_stocks: int = 600):
        self.data_dir = data_dir or ModelConfig.DATA_DIR
        self.max_stocks = max_stocks
        self.feat_tensor = None        # [num_stocks, N_features, T]
        self.raw_data_cache = None     # dict[str, Tensor]  各字段 [num_stocks, T]
        self.target_ret = None         # [num_stocks, T]
        self.stock_codes = []          # list[str]
        self.dates = None              # pd.DatetimeIndex
        self.valid_idx = None          # 验证集截止索引（不含）
        self.train_idx = None          # 训练集截止索引（不含）
        self.test_idx = None           # 测试集截止索引（不含）
        self.benchmark_ret = None      # [T] 基准指数日收益率

    def load_data(self):
        print("从 CSV 加载A股数据...")

        # 1. 读取成分股列表（当前 + 历史，消除幸存者偏差）
        const_path = os.path.join(self.data_dir, "constituents", "hs300.csv")
        if not os.path.exists(const_path):
            raise FileNotFoundError(
                f"未找到成分股列表 {const_path}\n"
                "请先运行: python data_download.py --token YOUR_TOKEN"
            )
        const_df = pd.read_csv(const_path)
        col = "con_code" if "con_code" in const_df.columns else "ts_code"
        current_codes = set(const_df[col].dropna().unique().tolist())

        # 读取历史成分股（全部时期并集）
        hist_path = os.path.join(self.data_dir, "constituents", "hs300_history.csv")
        hist_df = None
        all_codes = list(current_codes)
        if os.path.exists(hist_path):
            hist_df = pd.read_csv(hist_path)
            h_col = "con_code" if "con_code" in hist_df.columns else "ts_code"
            hist_codes = set(hist_df[h_col].dropna().unique().tolist())
            all_codes = sorted(current_codes | hist_codes)
            print(f"  当前成分股 {len(current_codes)} 只 + 历史 {len(hist_codes)} 只 → 合并 {len(all_codes)} 只")
        else:
            print(f"  [WARN] 未找到历史成分股文件 {hist_path}，仅使用当前成分股（存在幸存者偏差）")

        # 2. 逐个读取 CSV，合并为 master DataFrame
        daily_dir = os.path.join(self.data_dir, "daily")
        all_dfs = []
        loaded_codes = []
        for code in all_codes:
            csv_path = os.path.join(daily_dir, f"{code}.csv")
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            if df.empty or len(df) < 60:
                continue  # 数据太少无法计算因子
            df["ts_code"] = code
            all_dfs.append(df)
            loaded_codes.append(code)
            if len(loaded_codes) >= self.max_stocks:
                break

        if not all_dfs:
            raise ValueError("没有加载到任何股票数据")

        master = pd.concat(all_dfs, ignore_index=True)
        master = master.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        self.stock_codes = loaded_codes

        # 3. 不筛选公共日期：每只股票保留各自完整时间序列
        # 新股/停牌通过 NaN mask 在下游处理，截面算子仅作用于每日有数据的股票
        master = master.dropna(subset=["close"])

        # 4. Pivot 为 [stocks, T] 的 tensor
        # 未上市的日期自然为 NaN，ffill 不填充无前值的 NaN（新股不影响）
        # 停牌缺口由 ffill 用最近价格填充（合规）
        close_pivot = master.pivot(index="trade_date", columns="ts_code", values="close")
        close_pivot = close_pivot.sort_index()
        close_pivot = close_pivot[[c for c in loaded_codes if c in close_pivot.columns]]
        # 停牌日 close 为 NaN → 记录掩码
        suspended_mask = torch.tensor(
            close_pivot.isna().values.T, dtype=torch.bool, device=ModelConfig.DEVICE
        )

        def to_tensor(col_name):
            pivot = master.pivot(index="trade_date", columns="ts_code", values=col_name)
            pivot = pivot.sort_index()
            pivot = pivot[[c for c in loaded_codes if c in pivot.columns]]
            # 停牌/缺失用最近一个交易日数据填充
            pivot = pivot.ffill()
            return torch.tensor(pivot.values.T, dtype=torch.float32, device=ModelConfig.DEVICE)

        self.raw_data_cache = {
            "open":   to_tensor("open"),
            "high":   to_tensor("high"),
            "low":    to_tensor("low"),
            "close":  to_tensor("close"),
            "vol":    to_tensor("vol"),
            "amount": to_tensor("amount"),
            "suspended": suspended_mask,   # [N, T] bool, True=停牌
        }

        # turnover_rate 可能不存在于早期数据
        if "turnover_rate" in master.columns:
            self.raw_data_cache["turnover_rate"] = to_tensor("turnover_rate")
        else:
            print("[WARN] 数据中无 turnover_rate 列，用成交量代理计算换手率")
            T = self.raw_data_cache["close"].shape[1]
            N = self.raw_data_cache["close"].shape[0]
            # 用成交量 / 成交量20日均值 作为流动性代理（>1 视为有流动性）
            vol = self.raw_data_cache["vol"]
            vol_ma20 = torch.zeros_like(vol)
            if T >= 20:
                cum = torch.cumsum(vol, dim=1)
                vol_ma20[:, 19] = cum[:, 19] / 20.0
                vol_ma20[:, 20:] = (cum[:, 20:] - cum[:, :-20]) / 20.0
            vol_ratio = vol / (vol_ma20 + 1e-9)
            self.raw_data_cache["turnover_rate"] = vol_ratio

        self.dates = sorted(master["trade_date"].unique())

        # 4.5 按配置的日期范围裁剪
        start_dt = ModelConfig.DATA_START_DATE
        if start_dt:
            keep_indices = [i for i, d in enumerate(self.dates) if int(d) >= int(start_dt)]
            if keep_indices:
                s = keep_indices[0]
                self.dates = self.dates[s:]
                for key in self.raw_data_cache:
                    self.raw_data_cache[key] = self.raw_data_cache[key][:, s:]

        # 4.6 构建时点成分股掩码（消除幸存者偏差）
        if hist_df is not None:
            constituent_mask = self._build_constituent_mask(
                hist_df, loaded_codes, self.dates
            )
            self.raw_data_cache["constituent"] = constituent_mask
        else:
            # 无历史数据时退化为全 True（与旧行为一致）
            N, T = self.raw_data_cache["close"].shape
            self.raw_data_cache["constituent"] = torch.ones(
                N, T, dtype=torch.bool, device=ModelConfig.DEVICE
            )

        # 4.7 加载基准指数日收益率
        self._load_benchmark_returns()

        # 5. 计算因子
        self.feat_tensor = FeatureEngineer.compute_features(self.raw_data_cache)

        # 6. 计算 target_ret（open-to-open，T+1 合规）
        op = self.raw_data_cache["open"]
        suspended = self.raw_data_cache["suspended"]  # [N, T] 停牌掩码（ffill 前记录）
        op_next = torch.roll(op, -1, dims=1)
        op_next2 = torch.roll(op, -2, dims=1)
        self.target_ret = op_next2 / (op_next + 1e-9) - 1.0
        self.target_ret[:, -2:] = 0.0
        # 停牌日的 target_ret 设为 0（ffill 后价格不变→收益≈0，但显式清零更安全）
        self.target_ret[suspended] = 0.0
        # T+1 日也停牌的，open-to-open 收益无意义
        suspended_next = torch.roll(suspended, -1, dims=1)
        suspended_next[:, -1] = False
        self.target_ret[suspended_next] = 0.0
        # 复牌跳空极端收益截断（A股涨跌停 ±10%/±20%，超过 ±25% 视为异常）
        self.target_ret = self.target_ret.clamp(-0.25, 0.25)

        # 7. 验证/训练/测试三集切分
        self.valid_idx = 0
        self.train_idx = 0
        self.test_idx = 0
        valid_end = int(ModelConfig.VALID_END_DATE)
        train_end = int(ModelConfig.TRAIN_END_DATE)
        test_end = int(ModelConfig.TEST_END_DATE)
        for i, d in enumerate(self.dates):
            d_int = int(d)
            if d_int <= valid_end:
                self.valid_idx = i + 1
            if d_int <= train_end:
                self.train_idx = i + 1
            if d_int <= test_end:
                self.test_idx = i + 1

        # 边界校验
        T = len(self.dates)
        if self.valid_idx < 20:
            raise ValueError(
                f"验证集不足 20 个交易日（仅 {self.valid_idx} 天），"
                f"请检查 DATA_START_DATE / VALID_END_DATE 配置或数据完整性"
            )
        if self.train_idx <= self.valid_idx:
            raise ValueError(
                f"训练集为空（valid_idx={self.valid_idx}, train_idx={self.train_idx}），"
                f"请检查 VALID_END_DATE / TRAIN_END_DATE 配置或数据完整性"
            )
        if self.test_idx <= self.train_idx:
            raise ValueError(
                f"测试集为空（train_idx={self.train_idx}, test_idx={self.test_idx}），"
                f"请检查 TRAIN_END_DATE / TEST_END_DATE 配置或数据完整性"
            )

        N, T = self.raw_data_cache["close"].shape
        print(f"数据加载完成: {N} 只股票, {T} 个交易日, {self.feat_tensor.shape[1]} 个因子")
        print(f"  验证集: {self.dates[0]} ~ {self.dates[self.valid_idx-1]} ({self.valid_idx} 天)")
        print(f"  训练集: {self.dates[self.valid_idx]} ~ {self.dates[self.train_idx-1]} ({self.train_idx - self.valid_idx} 天)")
        print(f"  测试集: {self.dates[self.train_idx]} ~ {self.dates[self.test_idx-1]} ({self.test_idx - self.train_idx} 天)")
        print(f"  基准指数: {ModelConfig.BENCHMARK_INDEX}"
              f"{'（已加载）' if self.benchmark_ret is not None else '（未加载，使用等权基准）'}")

    def _load_benchmark_returns(self):
        """加载沪深300指数日收益率，与 self.dates 对齐。"""
        index_path = os.path.join(
            self.data_dir, "constituents", "hs300_index.csv"
        )
        if not os.path.exists(index_path):
            print("  [WARN] 未找到HS300指数数据，基准将使用等权组合")
            self.benchmark_ret = None
            return

        idx_df = pd.read_csv(index_path)
        if idx_df.empty or "trade_date" not in idx_df.columns:
            print("  [WARN] HS300指数数据为空，基准将使用等权组合")
            self.benchmark_ret = None
            return

        # 使用 pct_chg 列（日涨跌幅 %）或自行从 close 计算
        if "pct_chg" in idx_df.columns:
            idx_df["daily_ret"] = idx_df["pct_chg"].astype(float) / 100.0
        elif "close" in idx_df.columns:
            idx_df = idx_df.sort_values("trade_date")
            idx_df["daily_ret"] = idx_df["close"].astype(float).pct_change()
            idx_df = idx_df.dropna(subset=["daily_ret"])
        else:
            print("  [WARN] HS300指数数据无 pct_chg 或 close 列")
            self.benchmark_ret = None
            return

        # 与 self.dates 对齐：构建 date → daily_ret 映射
        idx_map = dict(zip(
            idx_df["trade_date"].astype(str),
            idx_df["daily_ret"].astype(float)
        ))
        T = len(self.dates)
        bench = torch.zeros(T, dtype=torch.float32, device=ModelConfig.DEVICE)
        matched = 0
        for i, d in enumerate(self.dates):
            ret = idx_map.get(str(d), 0.0)
            bench[i] = ret
            if str(d) in idx_map:
                matched += 1

        self.benchmark_ret = bench
        print(f"  HS300指数基准: {matched}/{T} 个交易日已匹配")

    def _build_constituent_mask(self, hist_df, loaded_codes, all_dates):
        """构建时点成分股掩码 [num_stocks, T]。

        对每个交易日，找到最近一次指数调整日，标记该股票是否为成分股。

        Args:
            hist_df: 历史成分股 DataFrame（含 trade_date, con_code 列）
            loaded_codes: 已加载的股票代码列表（与 tensor dim=0 对齐）
            all_dates: 所有交易日列表（与 tensor dim=1 对齐）
        Returns:
            constituent_mask: [N, T] bool tensor，True=该股票在该日期是沪深300成分股
        """
        h_col = "con_code" if "con_code" in hist_df.columns else "ts_code"

        # 按调整日分组：{rebalance_date: set(con_codes)}
        rebalance_dates = sorted(str(d) for d in hist_df["trade_date"].unique())
        rebalance_map = {}
        for rd in rebalance_dates:
            rebalance_map[rd] = set(
                hist_df.loc[hist_df["trade_date"].astype(str) == rd, h_col].tolist()
            )

        N = len(loaded_codes)
        T = len(all_dates)
        mask = torch.zeros(N, T, dtype=torch.bool, device=ModelConfig.DEVICE)

        # 向量化构建：预计算每个交易日的最近调整日索引，按调整日分组批量处理
        code_to_idx = {code: i for i, code in enumerate(loaded_codes)}
        from collections import defaultdict
        rd_to_t_indices = defaultdict(list)
        uncovered_t = []  # 没有匹配到调整日的交易日
        for t, date in enumerate(all_dates):
            idx = bisect.bisect_right(rebalance_dates, str(date)) - 1
            if idx >= 0:
                rd_to_t_indices[idx].append(t)
            else:
                uncovered_t.append(t)

        for rd_idx, t_indices in rd_to_t_indices.items():
            active_codes = rebalance_map[rebalance_dates[rd_idx]]
            for code in active_codes:
                if code in code_to_idx:
                    i = code_to_idx[code]
                    mask[i, t_indices] = True

        # 对没有匹配到调整日的交易日（数据不足），回退为全 True
        if uncovered_t:
            mask[:, uncovered_t] = True
            print(f"  [WARN] {len(uncovered_t)} 个交易日无调整日数据，回退为全部股票可用")

        # 打印诊断信息
        active_counts = mask.float().sum(dim=0)
        print(f"  PIT成分股掩码: 平均每日 {active_counts.mean():.0f} 只成分股 "
              f"(最少 {active_counts.min():.0f}, 最多 {active_counts.max():.0f})")

        return mask
