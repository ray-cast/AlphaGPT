import os
from datetime import datetime, timedelta

import pandas as pd
import torch
from .config import ModelConfig
from .factors import FeatureEngineer


# ------------------------------------------------------------------
#  股票池过滤工具
# ------------------------------------------------------------------

def is_excluded_board(code: str) -> bool:
    """判断股票是否属于排除板块（科创板/创业板/北交所）。"""
    c = code.split(".")[0]
    if ModelConfig.EXCLUDE_STAR and c.startswith("688"):
        return True
    if ModelConfig.EXCLUDE_GEM and (c.startswith("300") or c.startswith("301")):
        return True
    if ModelConfig.EXCLUDE_BSE and c.startswith("8"):
        return True
    return False


def is_st_stock(name: str) -> bool:
    """判断股票是否为 ST/*ST 股票（按当前名称）。"""
    if ModelConfig.EXCLUDE_ST:
        return "ST" in name.upper()
    return False


def build_ipo_mask(
    stock_codes: list[str],
    list_dates: list[str],
    trade_dates: list[str],
    min_days: int = None,
) -> torch.Tensor:
    """构建次新股过滤掩码 [N, T]。

    True = 已上市足够久（可交易），False = 次新股（排除）。
    """
    if min_days is None:
        min_days = ModelConfig.IPO_MIN_DAYS

    N = len(stock_codes)
    T = len(trade_dates)

    td_parsed = []
    for d in trade_dates:
        try:
            td_parsed.append(datetime.strptime(str(d), "%Y%m%d"))
        except ValueError:
            td_parsed.append(None)

    ipo_parsed = []
    for d in list_dates:
        try:
            ipo_parsed.append(datetime.strptime(str(d), "%Y%m%d"))
        except ValueError:
            ipo_parsed.append(None)

    mask = torch.zeros(N, T, dtype=torch.bool, device=ModelConfig.DEVICE)
    for i in range(N):
        if ipo_parsed[i] is None:
            mask[i, :] = True
            continue
        threshold = ipo_parsed[i] + timedelta(days=min_days)
        for t in range(T):
            if td_parsed[t] is not None and td_parsed[t] >= threshold:
                mask[i, t] = True

    return mask


def _limit_pct_for_code(code: str) -> float:
    """从股票代码前缀推断涨跌停百分比。"""
    c = code.split(".")[0]
    if c.startswith("300") or c.startswith("301") or c.startswith("688"):
        return ModelConfig.PRICE_LIMIT_GEM   # 0.20
    return ModelConfig.PRICE_LIMIT_MAIN       # 0.10


class AshareDataLoader:
    """从 CSV 文件加载全市场A股日线数据，转为 PyTorch tensors。

    首次加载后自动缓存为 .pt 文件，后续加载直接读缓存（快 ~50-100x）。
    CSV 文件有更新时缓存自动失效。
    """

    def __init__(self, data_dir: str = None, max_stocks: int = 3000):
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
        self.nan_mask = None           # [N, T] bool
        self.clean_feat_tensor = None  # [num_stocks, N_features, T]

    # ------------------------------------------------------------------
    #  缓存管理
    # ------------------------------------------------------------------

    def _cache_path(self) -> str:
        return os.path.join(self.data_dir, ".tensor_cache.pt")

    def _cache_is_valid(self) -> bool:
        """缓存文件存在且比所有 CSV 文件都新。"""
        cp = self._cache_path()
        if not os.path.exists(cp):
            return False
        cache_mtime = os.path.getmtime(cp)
        daily_dir = os.path.join(self.data_dir, "daily")
        for f in os.listdir(daily_dir):
            if f.endswith(".csv"):
                if os.path.getmtime(os.path.join(daily_dir, f)) > cache_mtime:
                    return False
        # constituents 目录也要检查
        const_dir = os.path.join(self.data_dir, "constituents")
        if os.path.isdir(const_dir):
            for f in os.listdir(const_dir):
                if f.endswith(".csv"):
                    if os.path.getmtime(os.path.join(const_dir, f)) > cache_mtime:
                        return False
        return True

    def _save_cache(self):
        """将原始 tensors 保存到缓存文件（不含因子/收益，因子每次重算）。"""
        cache = {
            "raw_data_cache": self.raw_data_cache,
            "stock_codes": self.stock_codes,
            "dates": self.dates,
            "benchmark_ret": self.benchmark_ret,
            "max_stocks": self.max_stocks,
        }
        torch.save(cache, self._cache_path())
        print("  原始数据缓存已保存，后续加载将跳过 CSV 解析")

    def _load_cache(self) -> bool:
        """尝试从缓存加载原始 tensors，成功返回 True。"""
        if not self._cache_is_valid():
            return False
        print("从缓存加载原始数据...")
        import time
        t0 = time.time()
        cache = torch.load(self._cache_path(), weights_only=False, map_location=ModelConfig.DEVICE)
        if cache.get("max_stocks") != self.max_stocks:
            print("  max_stocks 变化，缓存失效")
            return False
        self.raw_data_cache = cache["raw_data_cache"]
        self.stock_codes = cache["stock_codes"]
        self.dates = cache["dates"]
        self.benchmark_ret = cache["benchmark_ret"]
        elapsed = time.time() - t0
        N, T = self.raw_data_cache["close"].shape
        print(f"  缓存加载耗时: {elapsed:.1f}s ({N} 只股票, {T} 个交易日)")
        return True

    def _load_stock_basic(self) -> pd.DataFrame:
        """加载股票基本信息（上市日期、名称），用于过滤。"""
        path = os.path.join(self.data_dir, "constituents", "stock_basic.csv")
        if not os.path.exists(path):
            print("  [WARN] 未找到 stock_basic.csv，跳过 ST/次新股过滤")
            return pd.DataFrame()
        df = pd.read_csv(path, dtype={"list_date": str})
        return df

    def _get_eligible_codes(self, stock_basic_df: pd.DataFrame) -> tuple[list[str], list[str]]:
        """根据配置过滤股票代码，返回 (codes, list_dates)。"""
        daily_dir = os.path.join(self.data_dir, "daily")

        # 获取所有已有 CSV 的股票代码
        csv_files = [f for f in os.listdir(daily_dir) if f.endswith(".csv")]
        available_codes = sorted(f.replace(".csv", "") for f in csv_files)

        if stock_basic_df.empty:
            # 无元数据时，只做代码前缀过滤
            codes = [c for c in available_codes if not is_excluded_board(c)]
            return codes[: self.max_stocks], [""] * min(len(codes), self.max_stocks)

        # 构建元数据索引
        meta = {}
        for _, row in stock_basic_df.iterrows():
            meta[row["ts_code"]] = {
                "name": str(row.get("name", "")),
                "list_date": str(row.get("list_date", "")),
            }

        codes = []
        list_dates = []
        for code in available_codes:
            # 板块过滤
            if is_excluded_board(code):
                continue
            # ST 过滤（有元数据时）
            if code in meta:
                if is_st_stock(meta[code]["name"]):
                    continue
                list_dates.append(meta[code]["list_date"])
            else:
                list_dates.append("")
            codes.append(code)
            if len(codes) >= self.max_stocks:
                break

        return codes, list_dates

    def load_data(self):
        # ── 第一阶段：加载原始 tensors（缓存 或 CSV）──
        if not self._load_cache():
            self._load_from_csv()
            self._save_cache()

        # ── 第二阶段：因子 / 收益 / 切分（每次重算）──
        self._compute_derived()

    def _load_from_csv(self):
        """从 CSV 文件读取并转为原始 tensors（耗时步骤，可缓存）。"""
        print("从 CSV 加载全市场A股数据...")

        # 1. 加载股票基本信息（用于过滤）
        stock_basic_df = self._load_stock_basic()

        # 2. 获取过滤后的股票代码列表
        loaded_codes, list_dates = self._get_eligible_codes(stock_basic_df)
        if not loaded_codes:
            raise ValueError("过滤后无可用股票，请检查数据目录和过滤配置")
        print(f"  股票池: {len(loaded_codes)} 只（已排除创业板/科创板/北交所/ST）")

        # 3. 逐个读取 CSV，合并为 master DataFrame
        daily_dir = os.path.join(self.data_dir, "daily")
        all_dfs = []
        actual_codes = []
        actual_list_dates = []
        skipped = 0
        for i, code in enumerate(loaded_codes):
            csv_path = os.path.join(daily_dir, f"{code}.csv")
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            if df.empty or len(df) < 60:
                skipped += 1
                continue  # 数据太少无法计算因子
            df["ts_code"] = code
            all_dfs.append(df)
            actual_codes.append(code)
            actual_list_dates.append(list_dates[i])

        loaded_codes = actual_codes
        list_dates = actual_list_dates

        if not all_dfs:
            raise ValueError("没有加载到任何股票数据")
        if skipped:
            print(f"  跳过 {skipped} 只数据不足的股票")

        master = pd.concat(all_dfs, ignore_index=True)
        master = master.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        self.stock_codes = loaded_codes

        # 4. 不筛选公共日期：每只股票保留各自完整时间序列
        master = master.dropna(subset=["close"])

        # 5. Pivot 为 [stocks, T] 的 tensor
        close_pivot = master.pivot(index="trade_date", columns="ts_code", values="close")
        close_pivot = close_pivot.sort_index()
        close_pivot = close_pivot[[c for c in loaded_codes if c in close_pivot.columns]]
        suspended_mask = torch.tensor(
            close_pivot.isna().values.T, dtype=torch.bool, device=ModelConfig.DEVICE
        )

        def to_tensor(col_name):
            pivot = master.pivot(index="trade_date", columns="ts_code", values=col_name)
            pivot = pivot.sort_index()
            pivot = pivot[[c for c in loaded_codes if c in pivot.columns]]
            pivot = pivot.ffill()
            return torch.tensor(pivot.values.T, dtype=torch.float32, device=ModelConfig.DEVICE)

        self.raw_data_cache = {
            "open":   to_tensor("open"),
            "high":   to_tensor("high"),
            "low":    to_tensor("low"),
            "close":  to_tensor("close"),
            "vol":    to_tensor("vol"),
            "amount": to_tensor("amount"),
            "suspended": suspended_mask,
        }

        # turnover_rate
        if "turnover_rate" in master.columns:
            self.raw_data_cache["turnover_rate"] = to_tensor("turnover_rate")
        else:
            print("[WARN] 数据中无 turnover_rate 列，用成交量代理计算换手率")
            T = self.raw_data_cache["close"].shape[1]
            vol = self.raw_data_cache["vol"]
            vol_ma20 = torch.zeros_like(vol)
            if T >= 20:
                cum = torch.cumsum(vol, dim=1)
                vol_ma20[:, 19] = cum[:, 19] / 20.0
                vol_ma20[:, 20:] = (cum[:, 20:] - cum[:, :-20]) / 20.0
            vol_ratio = vol / (vol_ma20 + 1e-9)
            self.raw_data_cache["turnover_rate"] = vol_ratio

        # 基本面指标
        if "pe_ttm" in master.columns:
            self.raw_data_cache["pe_ttm"] = to_tensor("pe_ttm")
        if "pb" in master.columns:
            self.raw_data_cache["pb"] = to_tensor("pb")
        pe_ttm_t = self.raw_data_cache.get("pe_ttm")
        pb_t = self.raw_data_cache.get("pb")
        if pe_ttm_t is not None and pb_t is not None:
            self.raw_data_cache["roe"] = pb_t / pe_ttm_t

        # 6. 涨跌停掩码
        if "pre_close" in master.columns:
            pre_close_pivot = master.pivot(index="trade_date", columns="ts_code", values="pre_close")
            pre_close_pivot = pre_close_pivot.sort_index()
            pre_close_pivot = pre_close_pivot[[c for c in loaded_codes if c in pre_close_pivot.columns]]
            pre_close_t = torch.tensor(
                pre_close_pivot.values.T, dtype=torch.float32, device=ModelConfig.DEVICE
            )
        else:
            close_t = self.raw_data_cache["close"]
            pre_close_t = torch.zeros_like(close_t)
            pre_close_t[:, 1:] = close_t[:, :-1]
            pre_close_t[:, 0] = close_t[:, 0]

        close_t = self.raw_data_cache["close"]
        pct_change = (close_t - pre_close_t) / (pre_close_t.abs() + 1e-8)
        thresholds = torch.tensor(
            [_limit_pct_for_code(code) for code in loaded_codes],
            dtype=torch.float32, device=ModelConfig.DEVICE,
        ).unsqueeze(1)  # [N, 1]

        # ST 检测：主板股票若反复在 ±5% 附近收于极值价（最高/最低），疑似 ST
        high_t = self.raw_data_cache["high"]
        low_t = self.raw_data_cache["low"]
        at_ceil_5 = (pct_change >= 0.045) & ((high_t - close_t) / (close_t + 1e-8) < 0.002)
        at_floor_5 = (pct_change <= -0.045) & ((close_t - low_t) / (close_t + 1e-8) < 0.002)
        st_score = (at_ceil_5 | at_floor_5).float().sum(dim=1)  # [N]
        is_main = (thresholds.squeeze() == ModelConfig.PRICE_LIMIT_MAIN)  # [N]
        is_st = (st_score >= 3) & is_main
        if is_st.any():
            thresholds[is_st] = ModelConfig.PRICE_LIMIT_ST

        limit_up = pct_change >= (thresholds - 0.005)
        limit_down = pct_change <= (-thresholds + 0.005)
        limit_up[:, 0] = False
        limit_down[:, 0] = False
        limit_up[suspended_mask] = False
        limit_down[suspended_mask] = False
        self.raw_data_cache["limit_up"] = limit_up
        self.raw_data_cache["limit_down"] = limit_down

        self.dates = sorted(master["trade_date"].unique())

        # 7. 按配置的日期范围裁剪
        start_dt = ModelConfig.DATA_START_DATE
        if start_dt:
            keep_indices = [i for i, d in enumerate(self.dates) if int(d) >= int(start_dt)]
            if keep_indices:
                s = keep_indices[0]
                self.dates = self.dates[s:]
                for key in self.raw_data_cache:
                    self.raw_data_cache[key] = self.raw_data_cache[key][:, s:]

        # 7.1 印花税时间分段（2023-08-28 由千1降至千0.5）
        stamp_tax = torch.full(
            (len(self.dates),), ModelConfig.STAMP_TAX_RATE_OLD,
            dtype=torch.float32, device=ModelConfig.DEVICE,
        )
        for i, d in enumerate(self.dates):
            if int(d) >= int(ModelConfig.STAMP_TAX_CHANGE_DATE):
                stamp_tax[i:] = ModelConfig.STAMP_TAX_RATE_NEW
                break
        self.raw_data_cache["stamp_tax_rate"] = stamp_tax  # [T]

        # 8. 构建次新股掩码（替代原成分股掩码，消除新股影响）
        ipo_mask = build_ipo_mask(loaded_codes, list_dates, self.dates)
        self.raw_data_cache["ipo_ok"] = ipo_mask  # [N, T] True=已上市足够久

        # 9. 加载基准指数日收益率
        self._load_benchmark_returns()

    def _compute_derived(self):
        """从原始 tensors 计算因子、收益率、数据集切分（每次重算）。"""
        # 10. 计算因子
        self.feat_tensor = FeatureEngineer.compute_features(self.raw_data_cache)

        # 10.1 预计算 NaN mask 和清洗版 feat
        self.nan_mask = torch.isnan(self.feat_tensor).any(dim=1)   # [N, T] bool
        self.clean_feat_tensor = self.feat_tensor.nan_to_num(nan=0.0)

        # 11. 计算 target_ret（open-to-open，T+1 合规）
        op = self.raw_data_cache["open"]
        suspended = self.raw_data_cache["suspended"]
        op_next = torch.roll(op, -1, dims=1)
        op_next2 = torch.roll(op, -2, dims=1)
        self.target_ret = op_next2 / (op_next + 1e-9) - 1.0
        self.target_ret[:, -2:] = 0.0
        self.target_ret[suspended] = 0.0
        suspended_next = torch.roll(suspended, -1, dims=1)
        suspended_next[:, -1] = False
        self.target_ret[suspended_next] = 0.0
        self.target_ret = self.target_ret.clamp(-0.25, 0.25)

        # 12. 验证/训练/测试三集切分
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
        ipo_mask = self.raw_data_cache.get("ipo_ok")
        ipo_ok_pct = ipo_mask.float().mean() * 100 if ipo_mask is not None else 0.0
        print(f"数据加载完成: {N} 只股票, {T} 个交易日, {self.feat_tensor.shape[1]} 个因子")
        print(f"  次新股掩码: 平均 {ipo_ok_pct:.1f}% 的股票-日通过（上市≥{ModelConfig.IPO_MIN_DAYS}天）")
        print(f"  验证集: {self.dates[0]} ~ {self.dates[self.valid_idx-1]} ({self.valid_idx} 天)")
        print(f"  训练集: {self.dates[self.valid_idx]} ~ {self.dates[self.train_idx-1]} ({self.train_idx - self.valid_idx} 天)")
        print(f"  测试集: {self.dates[self.train_idx]} ~ {self.dates[self.test_idx-1]} ({self.test_idx - self.train_idx} 天)")
        print(f"  基准指数: {ModelConfig.BENCHMARK_INDEX}"
              f"{'（已加载）' if self.benchmark_ret is not None else '（未加载，使用等权基准）'}")

    def _load_benchmark_returns(self):
        """加载沪深300指数日收益率，与 self.dates 对齐。"""
        index_path = os.path.join(
            self.data_dir, "constituents", "benchmark_index.csv"
        )
        if not os.path.exists(index_path):
            print("  [WARN] 未找到基准指数数据，基准将使用等权组合")
            self.benchmark_ret = None
            return

        idx_df = pd.read_csv(index_path)
        if idx_df.empty or "trade_date" not in idx_df.columns:
            print("  [WARN] 基准指数数据为空，基准将使用等权组合")
            self.benchmark_ret = None
            return

        if "pct_chg" in idx_df.columns:
            idx_df["daily_ret"] = idx_df["pct_chg"].astype(float) / 100.0
        elif "close" in idx_df.columns:
            idx_df = idx_df.sort_values("trade_date")
            idx_df["daily_ret"] = idx_df["close"].astype(float).pct_change()
            idx_df = idx_df.dropna(subset=["daily_ret"])
        else:
            print("  [WARN] 基准指数数据无 pct_chg 或 close 列")
            self.benchmark_ret = None
            return

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
        print(f"  基准指数: {matched}/{T} 个交易日已匹配")
