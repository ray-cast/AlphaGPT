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
        self.split_idx = None          # 80/20 训练/测试切分点

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
        def to_tensor(col_name):
            pivot = master.pivot(index="trade_date", columns="ts_code", values=col_name)
            pivot = pivot.sort_index()
            # 按加载顺序排列列
            pivot = pivot[[c for c in loaded_codes if c in pivot.columns]]
            # 停牌/缺失保留 NaN，不填 0（避免产生虚假收益率）
            pivot = pivot.ffill()
            return torch.tensor(pivot.values.T, dtype=torch.float32, device=ModelConfig.DEVICE)

        self.raw_data_cache = {
            "open":   to_tensor("open"),
            "high":   to_tensor("high"),
            "low":    to_tensor("low"),
            "close":  to_tensor("close"),
            "vol":    to_tensor("vol"),
            "amount": to_tensor("amount"),
        }

        # turnover_rate 可能不存在于早期数据
        if "turnover_rate" in master.columns:
            self.raw_data_cache["turnover_rate"] = to_tensor("turnover_rate")
        else:
            T = self.raw_data_cache["close"].shape[1]
            N = self.raw_data_cache["close"].shape[0]
            self.raw_data_cache["turnover_rate"] = torch.zeros(
                (N, T), dtype=torch.float32, device=ModelConfig.DEVICE
            )

        self.dates = sorted(common_dates)

        # 4.5 按配置的日期范围裁剪
        start_dt = ModelConfig.TRAIN_START_DATE
        end_dt = ModelConfig.TRAIN_END_DATE
        if start_dt or end_dt:
            mask = [True] * len(self.dates)
            for i, d in enumerate(self.dates):
                d_int = int(d)
                if start_dt and d_int < int(start_dt):
                    mask[i] = False
                if end_dt and d_int > int(end_dt):
                    mask[i] = False
            keep_indices = [i for i, m in enumerate(mask) if m]
            if keep_indices:
                s, e = keep_indices[0], keep_indices[-1] + 1
                self.dates = self.dates[s:e]
                for key in self.raw_data_cache:
                    self.raw_data_cache[key] = self.raw_data_cache[key][:, s:e]

        # 5. 计算因子
        self.feat_tensor = FeatureEngineer.compute_features(self.raw_data_cache)

        # 6. 计算 target_ret（open-to-open，T+1 合规）
        op = self.raw_data_cache["open"]
        op_next = torch.roll(op, -1, dims=1)
        op_next2 = torch.roll(op, -2, dims=1)
        self.target_ret = op_next2 / (op_next + 1e-9) - 1.0
        self.target_ret[:, -2:] = 0.0
        # 停牌日（open 为 NaN）的 target_ret 设为 0，排除虚假收益
        suspended = torch.isnan(op)
        self.target_ret[suspended] = 0.0
        # 下一个交易日也停牌的，其 target_ret 也无意义
        suspended_next = torch.isnan(op_next)
        self.target_ret[suspended_next] = 0.0
        # 复牌跳空极端收益截断（A股涨跌停 ±10%/±20%，超过 ±25% 视为异常）
        self.target_ret = self.target_ret.clamp(-0.25, 0.25)

        # 7. 训练/测试切分
        self.split_idx = int(len(self.dates) * ModelConfig.TRAIN_RATIO)

        N, T = self.raw_data_cache["close"].shape
        print(f"数据加载完成: {N} 只股票, {T} 个交易日, {self.feat_tensor.shape[1]} 个因子")
        print(f"  训练集: {self.dates[0]} ~ {self.dates[self.split_idx-1]}")
        print(f"  测试集: {self.dates[self.split_idx]} ~ {self.dates[-1]}")
