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
    """构建次新股过滤掩码 [N, T]（向量化版本）。

    True = 已上市足够久（可交易），False = 次新股（排除）。
    """
    if min_days is None:
        min_days = ModelConfig.IPO_MIN_DAYS

    N = len(stock_codes)
    T = len(trade_dates)

    # 向量化日期解析：将 YYYYMMDD 整数 → 天数 ordinal
    def _to_ordinals(dates, default=-1):
        out = []
        for d in dates:
            try:
                out.append(datetime.strptime(str(d), "%Y%m%d").toordinal())
            except ValueError:
                out.append(default)
        return out

    td_ord = torch.tensor(_to_ordinals(trade_dates, -1), dtype=torch.int64)           # [T]
    ipo_ord = torch.tensor(_to_ordinals(list_dates, -1), dtype=torch.int64)            # [N]
    no_ipo = ipo_ord < 0                                                                # [N] 无上市日期

    # 广播比较: threshold[N,1] >= td_ord[1,T] → mask[N,T]
    threshold = ipo_ord + min_days                                                      # [N]
    valid_td = td_ord >= 0                                                              # [T]
    mask = (threshold.unsqueeze(1) <= td_ord.unsqueeze(0)) & valid_td.unsqueeze(0)     # [N,T]
    mask[no_ipo] = True  # 无上市日期的股票视为已上市

    return mask.to(device=ModelConfig.DEVICE)


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
        self.valid_start = None        # 验证集起始索引（含）
        self.valid_end = None          # 验证集截止索引（不含）= 训练集起始
        self.train_start = None        # 训练集起始索引（含）= valid_end
        self.train_end = None          # 训练集截止索引（不含）= 测试集起始
        self.test_start = None         # 测试集起始索引（含）= train_end
        self.test_end = None           # 测试集截止索引（不含）
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

        # 5. 单次 pivot 提取全部列（避免重复 pivot/sort_index/列选择）
        code_order = [c for c in loaded_codes]

        # 收集所有需要 pivot 的列
        ohlcv_cols = ["open", "high", "low", "close", "vol", "amount"]
        optional_cols = ["turnover_rate", "pe_ttm", "pb", "total_mv", "pre_close", "pct_chg"]
        all_pivot_cols = [c for c in ohlcv_cols + optional_cols if c in master.columns]

        # 一次性 set_index + unstack 代替多次 pivot
        indexed = master.set_index(["trade_date", "ts_code"])[all_pivot_cols]
        pivoted = indexed.unstack("ts_code")  # MultiIndex columns: (col_name, ts_code)
        pivoted = pivoted.sort_index()         # 按日期排序
        # 统一列顺序
        pivoted = pivoted.reorder_levels([1, 0], axis=1)
        pivoted = pivoted[code_order] if all(c in pivoted.columns.get_level_values(0) for c in code_order) else pivoted

        dev = ModelConfig.DEVICE

        def _col_to_tensor(col_name, ffill=True):
            """从预计算的 pivoted 表提取单列 [N, T] tensor。"""
            if col_name not in all_pivot_cols:
                return None
            sub = pivoted.xs(col_name, level=1, axis=1)
            # 对齐列顺序
            sub = sub[[c for c in code_order if c in sub.columns]]
            if ffill:
                sub = sub.ffill()
            return torch.tensor(sub.values.T, dtype=torch.float32, device=dev)

        # close 不做 ffill，用于构建 suspended_mask
        close_sub = pivoted.xs("close", level=1, axis=1)
        close_sub = close_sub[[c for c in code_order if c in close_sub.columns]]
        suspended_mask = torch.tensor(
            close_sub.isna().values.T, dtype=torch.bool, device=dev
        )

        self.raw_data_cache = {
            "open":      _col_to_tensor("open"),
            "high":      _col_to_tensor("high"),
            "low":       _col_to_tensor("low"),
            "close":     _col_to_tensor("close"),
            "vol":       _col_to_tensor("vol"),
            "amount":    _col_to_tensor("amount"),
            "suspended": suspended_mask,
        }

        # turnover_rate
        if "turnover_rate" in all_pivot_cols:
            self.raw_data_cache["turnover_rate"] = _col_to_tensor("turnover_rate")
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
        for key in ("pe_ttm", "pb", "total_mv"):
            t = _col_to_tensor(key)
            if t is not None:
                self.raw_data_cache[key] = t
        pe_ttm_t = self.raw_data_cache.get("pe_ttm")
        pb_t = self.raw_data_cache.get("pb")
        if pe_ttm_t is not None and pb_t is not None:
            self.raw_data_cache["roe"] = pb_t / pe_ttm_t

        # 6. 涨跌停掩码
        pre_close_t = _col_to_tensor("pre_close")
        if pre_close_t is None:
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

        # 6.5 后复权：用 pct_chg 构造除权除息不变的调整价格
        #     adj_close[t] = close[0] * cumprod(1 + pct_chg/100)
        #     adj_open/high/low = adj_close * (x / close)  日内比率不受除权影响
        #     必须在涨跌停检测之后执行（涨跌停依赖原始价格）
        if "pct_chg" in all_pivot_cols:
            pct_chg_t = _col_to_tensor("pct_chg") / 100.0     # [N, T] 日收益率
            close_raw = self.raw_data_cache["close"]           # [N, T]（已 ffill）
            # _col_to_tensor 做了 ffill，停牌日的 pct_chg/close 不是 NaN 而是前值，
            # 必须用 suspended_mask 显式跳过，否则停牌日会错误增长
            skip = torch.isnan(pct_chg_t) | torch.isnan(close_raw) | suspended_mask
            ret_safe = torch.where(skip, torch.zeros_like(pct_chg_t), pct_chg_t)
            growth = 1.0 + ret_safe
            growth[:, 0] = 1.0                                 # 首日为基线
            cum_growth = torch.cumprod(growth, dim=1)          # [N, T]
            adj_close = close_raw[:, 0:1] * cum_growth         # 后复权收盘价
            # 用日内比率推导 adj_open/high/low
            adj_open  = adj_close * (self.raw_data_cache["open"]  / (close_raw + 1e-9))
            adj_high  = adj_close * (self.raw_data_cache["high"]  / (close_raw + 1e-9))
            adj_low   = adj_close * (self.raw_data_cache["low"]   / (close_raw + 1e-9))
            # 停牌日保留 ffill 值（维持 MA/EMA 计算稳定），仅还原原始数据缺失的 NaN
            orig_nan = torch.isnan(pct_chg_t) | torch.isnan(close_raw)
            _nan = torch.full_like(adj_close, float('nan'))
            self.raw_data_cache["close"] = torch.where(orig_nan, _nan, adj_close)
            self.raw_data_cache["open"]  = torch.where(orig_nan, _nan, adj_open)
            self.raw_data_cache["high"]  = torch.where(orig_nan, _nan, adj_high)
            self.raw_data_cache["low"]   = torch.where(orig_nan, _nan, adj_low)
            print("  后复权: 已用 pct_chg 构造调整价格（OHLC）")

        self.dates = sorted(master["trade_date"].unique())

        # 7. 按配置的日期范围裁剪
        start_dt = ModelConfig.VALID_START_DATE
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

        # 10.0 停牌日因子置 NaN：ffill 导致停牌日产生虚假因子值（RET=0, VOL_CHG=0 等），
        #     必须显式标记为无效，防止模型学习到停牌日的伪造信号
        suspended = self.raw_data_cache.get("suspended")
        if suspended is not None:
            # [N, T] → [N, 1, T] broadcast over feature dim → [N, F, T]
            self.feat_tensor.masked_fill_(suspended.unsqueeze(1), float('nan'))

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
        #    每个区间为半开区间 [start, end)
        self.valid_start = 0
        self.valid_end = 0
        self.train_start = 0
        self.train_end = 0
        self.test_start = 0
        self.test_end = 0
        _valid_start = int(ModelConfig.VALID_START_DATE)
        _valid_end = int(ModelConfig.VALID_END_DATE)
        _train_start = int(ModelConfig.TRAIN_START_DATE)
        _train_end = int(ModelConfig.TRAIN_END_DATE)
        _test_start = int(ModelConfig.TEST_START_DATE)
        _test_end = int(ModelConfig.TEST_END_DATE)
        for i, d in enumerate(self.dates):
            d_int = int(d)
            if d_int < _valid_start:
                self.valid_start = i + 1
            if d_int <= _valid_end:
                self.valid_end = i + 1
            if d_int < _train_start:
                self.train_start = i + 1
            if d_int <= _train_end:
                self.train_end = i + 1
            if d_int < _test_start:
                self.test_start = i + 1
            if d_int <= _test_end:
                self.test_end = i + 1

        # 边界校验
        T = len(self.dates)
        if self.valid_end < 20:
            raise ValueError(
                f"验证集不足 20 个交易日（仅 {self.valid_end} 天），"
                f"请检查 VALID_START_DATE / VALID_END_DATE 配置或数据完整性"
            )
        if self.train_end <= self.valid_end:
            raise ValueError(
                f"训练集为空（valid_end={self.valid_end}, train_end={self.train_end}），"
                f"请检查 VALID_END_DATE / TRAIN_END_DATE 配置或数据完整性"
            )
        if self.test_end <= self.train_end:
            raise ValueError(
                f"测试集为空（train_end={self.train_end}, test_end={self.test_end}），"
                f"请检查 TRAIN_END_DATE / TEST_END_DATE 配置或数据完整性"
            )

        N, T = self.raw_data_cache["close"].shape
        ipo_mask = self.raw_data_cache.get("ipo_ok")
        ipo_ok_pct = ipo_mask.float().mean() * 100 if ipo_mask is not None else 0.0
        print(f"数据加载完成: {N} 只股票, {T} 个交易日, {self.feat_tensor.shape[1]} 个因子")
        print(f"  次新股掩码: 平均 {ipo_ok_pct:.1f}% 的股票-日通过（上市≥{ModelConfig.IPO_MIN_DAYS}天）")
        print(f"  验证集: {self.dates[self.valid_start]} ~ {self.dates[self.valid_end-1]} ({self.valid_end - self.valid_start} 天)")
        print(f"  训练集: {self.dates[self.train_start]} ~ {self.dates[self.train_end-1]} ({self.train_end - self.train_start} 天)")
        print(f"  测试集: {self.dates[self.test_start]} ~ {self.dates[self.test_end-1]} ({self.test_end - self.test_start} 天)")
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
