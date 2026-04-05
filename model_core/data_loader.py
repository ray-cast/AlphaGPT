import os
import glob
import pandas as pd
import torch
from .config import ModelConfig
from .factors import FeatureEngineer


class AshareDataLoader:
    """从 CSV 文件加载沪深300成分股日线数据，转为 PyTorch tensors。"""

    def __init__(self, data_dir: str = None, max_stocks: int = 300):
        self.data_dir = data_dir or ModelConfig.DATA_DIR
        self.max_stocks = max_stocks
        self.feat_tensor = None        # [num_stocks, N_features, T]
        self.raw_data_cache = None     # dict[str, Tensor]  各字段 [num_stocks, T]
        self.target_ret = None         # [num_stocks, T]
        self.stock_codes = []          # list[str]
        self.dates = None              # pd.DatetimeIndex
        self.train_idx = None          # 训练集截止索引（不含）
        self.test_idx = None           # 测试集截止索引（不含）

    def load_data(self):
        print("从 CSV 加载A股数据...")

        # 1. 读取成分股列表
        const_path = os.path.join(self.data_dir, "constituents", "hs300.csv")
        if not os.path.exists(const_path):
            raise FileNotFoundError(
                f"未找到成分股列表 {const_path}\n"
                "请先运行: python data_download.py --token YOUR_TOKEN"
            )
        const_df = pd.read_csv(const_path)
        col = "con_code" if "con_code" in const_df.columns else "ts_code"
        all_codes = const_df[col].dropna().unique().tolist()

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

        # 3. 找到交易日：使用至少 80% 股票都有的日期（避免因少数新股导致交集过小）
        # 先去掉停牌导致的价格 NaN 行，避免停牌股票被误计入日期统计
        master = master.dropna(subset=["close"])
        all_unique_dates = sorted(master["trade_date"].unique())
        date_count = {}
        for code in loaded_codes:
            sub = master[master["ts_code"] == code]
            for d in sub["trade_date"].tolist():
                date_count[d] = date_count.get(d, 0) + 1
        threshold = len(loaded_codes) * 0.8
        common_dates = [d for d in all_unique_dates if date_count.get(d, 0) >= threshold]

        # 过滤 master 只保留公共日期
        master = master[master["trade_date"].isin(common_dates)]

        # 4. Pivot 为 [stocks, T] 的 tensor
        # 先用 close 构建停牌掩码（ffill 之前），再统一做前向填充
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

        self.dates = sorted(common_dates)

        # 4.5 按配置的日期范围裁剪
        start_dt = ModelConfig.DATA_START_DATE
        if start_dt:
            keep_indices = [i for i, d in enumerate(self.dates) if int(d) >= int(start_dt)]
            if keep_indices:
                s = keep_indices[0]
                self.dates = self.dates[s:]
                for key in self.raw_data_cache:
                    self.raw_data_cache[key] = self.raw_data_cache[key][:, s:]

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

        # 7. 训练/测试/验证三集切分
        self.train_idx = 0  # 默认值
        self.test_idx = 0   # 默认值
        train_end = int(ModelConfig.TRAIN_END_DATE)
        test_end = int(ModelConfig.TEST_END_DATE)
        for i, d in enumerate(self.dates):
            d_int = int(d)
            if d_int <= train_end:
                self.train_idx = i + 1
            if d_int <= test_end:
                self.test_idx = i + 1

        # 边界校验
        T = len(self.dates)
        if self.train_idx < 60:
            raise ValueError(
                f"训练集不足 60 个交易日（仅 {self.train_idx} 天），"
                f"请检查 DATA_START_DATE / TRAIN_END_DATE 配置或数据完整性"
            )
        if self.test_idx <= self.train_idx:
            raise ValueError(
                f"测试集为空（train_idx={self.train_idx}, test_idx={self.test_idx}），"
                f"请检查 TRAIN_END_DATE / TEST_END_DATE 配置或数据完整性"
            )
        if T - self.test_idx < 20:
            print(f"[WARN] 验证集仅 {T - self.test_idx} 个交易日，统计指标可能不可靠")

        N, T = self.raw_data_cache["close"].shape
        print(f"数据加载完成: {N} 只股票, {T} 个交易日, {self.feat_tensor.shape[1]} 个因子")
        if self.train_idx > 0 and self.test_idx > self.train_idx:
            print(f"  训练集: {self.dates[0]} ~ {self.dates[self.train_idx-1]} ({self.train_idx} 天)")
            print(f"  测试集: {self.dates[self.train_idx]} ~ {self.dates[self.test_idx-1]} ({self.test_idx - self.train_idx} 天)")
            print(f"  验证集: {self.dates[self.test_idx]} ~ {self.dates[-1]} ({T - self.test_idx} 天)")
