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

        # 3. 找到公共交易日（所有股票的交集）
        date_sets = []
        for code in loaded_codes:
            sub = master[master["ts_code"] == code]
            date_sets.append(set(sub["trade_date"].tolist()))
        common_dates = sorted(set.intersection(*date_sets))

        # 如果公共日期太少，用每个股票各自的日期，后续 forward-fill
        if len(common_dates) < 100:
            common_dates = sorted(master["trade_date"].unique())

        # 过滤 master 只保留公共日期
        master = master[master["trade_date"].isin(common_dates)]

        # 4. Pivot 为 [stocks, T] 的 tensor
        def to_tensor(col_name):
            pivot = master.pivot(index="trade_date", columns="ts_code", values=col_name)
            pivot = pivot.sort_index()
            # 按加载顺序排列列
            pivot = pivot[[c for c in loaded_codes if c in pivot.columns]]
            pivot = pivot.ffill().fillna(0.0)
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

        # 5. 计算因子
        self.feat_tensor = FeatureEngineer.compute_features(self.raw_data_cache)

        # 6. 计算 target_ret（open-to-open，T+1 合规）
        op = self.raw_data_cache["open"]
        op_next = torch.roll(op, -1, dims=1)
        op_next2 = torch.roll(op, -2, dims=1)
        self.target_ret = op_next2 / (op_next + 1e-9) - 1.0
        self.target_ret[:, -2:] = 0.0

        # 7. 训练/测试切分
        self.split_idx = int(len(self.dates) * 0.8)

        N, T = self.raw_data_cache["close"].shape
        print(f"数据加载完成: {N} 只股票, {T} 个交易日, {self.feat_tensor.shape[1]} 个因子")
        print(f"  训练集: {self.dates[0]} ~ {self.dates[self.split_idx-1]}")
        print(f"  测试集: {self.dates[self.split_idx]} ~ {self.dates[-1]}")
