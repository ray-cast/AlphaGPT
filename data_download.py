"""
全A股（主板）数据下载脚本

过滤规则:
    - 排除科创板 (688xxx)
    - 排除创业板 (300xxx, 301xxx)
    - 排除北交所 (8xxxxx)
    - 排除 ST/*ST 股票
    - 排除次新股（上市不足 250 个自然日）

用法:
    python data_download.py [--start 20150101] [--end 20260404]

    Token 优先从 .env 文件读取 TUSHARE_TOKEN，也可通过 --token 参数覆盖。

数据目录结构:
    data/
      constituents/
        stock_basic.csv         # 全市场股票基本信息（含上市日期、名称）
        benchmark_index.csv     # 基准指数日线（中证全指）
      daily/
        000001.SZ.csv          # 每只股票一个CSV文件
        600519.SH.csv
        ...
"""

import os
import sys
import time
import argparse
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

try:
    import tushare as ts
except ImportError:
    print("请先安装 tushare: pip install tushare")
    sys.exit(1)

# 加载 .env 文件中的环境变量
load_dotenv(override=True)


class TushareDownloader:
    def __init__(self, token: str, data_dir: str = "data"):
        self.pro = ts.pro_api(token)
        self.data_dir = data_dir
        self.const_dir = os.path.join(data_dir, "constituents")
        self.daily_dir = os.path.join(data_dir, "daily")
        os.makedirs(self.const_dir, exist_ok=True)
        os.makedirs(self.daily_dir, exist_ok=True)

    # ------------------------------------------------------------------
    #  全市场股票列表
    # ------------------------------------------------------------------
    def fetch_stock_basic(self) -> pd.DataFrame:
        """获取全A股股票基本信息，缓存到 CSV。

        包含字段: ts_code, name, area, industry, market, list_date
        """
        cache_path = os.path.join(self.const_dir, "stock_basic.csv")
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path, dtype={"list_date": str})
            print(f"从缓存加载股票列表: {len(df)} 只")
            return df

        df = self.pro.stock_basic(
            exchange="",
            list_status="L",
            fields="ts_code,name,area,industry,market,list_date",
        )
        df["list_date"] = df["list_date"].astype(str)
        df.to_csv(cache_path, index=False, encoding="utf-8-sig")
        print(f"股票列表已保存: {cache_path} ({len(df)} 只)")
        return df

    def get_filtered_codes(
        self,
        exclude_gem: bool = True,
        exclude_star: bool = True,
        exclude_bse: bool = True,
        exclude_st: bool = True,
        ipo_min_days: int = 250,
    ) -> list[str]:
        """获取过滤后的股票代码列表（用于下载）。

        过滤规则:
            - 科创板 (688xxx)
            - 创业板 (300xxx, 301xxx)
            - 北交所 (8xxxxx)
            - ST/*ST 股票（按当前名称判断）
            - 次新股（上市不足 ipo_min_days 个自然日）
        """
        df = self.fetch_stock_basic()
        before = len(df)

        # 提取纯代码前缀
        codes = df["ts_code"].str.split(".").str[0]
        names = df["name"].fillna("").str.upper()
        list_dates = df["list_date"].fillna("").astype(str)

        # 板块过滤
        keep = pd.Series(True, index=df.index)
        if exclude_star:
            keep &= ~codes.str.startswith("688")
        if exclude_gem:
            keep &= ~codes.str.startswith("300") & ~codes.str.startswith("301")
        if exclude_bse:
            keep &= ~codes.str.startswith("8")
        # ST 过滤
        if exclude_st:
            keep &= ~names.str.contains("ST")
        # 次新股过滤
        valid_date = list_dates.str.len() == 8
        if ipo_min_days > 0:
            today = datetime.now()
            ipo_dates = pd.to_datetime(list_dates.where(valid_date), format="%Y%m%d", errors="coerce")
            days_since = (today - ipo_dates).dt.days
            keep &= ~(valid_date & (days_since < ipo_min_days))

        filtered = df.loc[keep, "ts_code"].tolist()
        n_excluded = before - len(filtered)
        print(f"股票过滤: {before} → {len(filtered)} 只 (排除 {n_excluded} 只)")
        return sorted(filtered)

    # ------------------------------------------------------------------
    #  指数日线（基准）
    # ------------------------------------------------------------------
    def fetch_index_daily(self, start_date: str = "20160101", end_date: str = None,
                          index_code: str = "000985.SH"):
        """下载指数日线行情，作为回测基准。默认中证全指。"""
        if end_date is None:
            end_date = time.strftime("%Y%m%d")
        cache_path = os.path.join(self.const_dir, "benchmark_index.csv")

        actual_start = start_date
        append_mode = False
        if os.path.exists(cache_path):
            existing = pd.read_csv(cache_path, dtype={"trade_date": str})
            if not existing.empty:
                last_date = str(existing["trade_date"].max())
                actual_start = last_date
                append_mode = True
                if last_date >= end_date:
                    print(f"指数数据已是最新 ({last_date})")
                    return existing

        try:
            df = self.pro.index_daily(
                ts_code=index_code,
                start_date=actual_start,
                end_date=end_date,
            )
        except Exception as e:
            print(f"  [WARN] 指数获取失败: {e}")
            if os.path.exists(cache_path):
                return pd.read_csv(cache_path)
            return pd.DataFrame()

        if df.empty:
            return pd.read_csv(cache_path) if os.path.exists(cache_path) else df

        df["trade_date"] = df["trade_date"].astype(str)
        df = df.sort_values("trade_date").reset_index(drop=True)

        if append_mode and os.path.exists(cache_path):
            existing = pd.read_csv(cache_path, dtype={"trade_date": str})
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["trade_date"], keep="last")
            combined = combined.sort_values("trade_date").reset_index(drop=True)
            combined.to_csv(cache_path, index=False, encoding="utf-8-sig")
            print(f"指数已更新: {cache_path} ({len(combined)} 行)")
            return combined
        else:
            df.to_csv(cache_path, index=False, encoding="utf-8-sig")
            print(f"指数已保存: {cache_path} ({len(df)} 行)")
            return df

    # ------------------------------------------------------------------
    #  单只股票日线数据
    # ------------------------------------------------------------------
    def fetch_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取单只股票日线数据，支持增量更新。"""
        csv_path = os.path.join(self.daily_dir, f"{ts_code}.csv")

        # 增量逻辑：如果文件已存在，只获取新增数据
        actual_start = start_date
        append_mode = False
        if os.path.exists(csv_path):
            existing = pd.read_csv(csv_path, dtype={"trade_date": str})
            if not existing.empty:
                last_date = str(existing["trade_date"].max())
                # 从下一个交易日开始
                actual_start = last_date
                append_mode = True
                # 如果数据已是最新则跳过
                if last_date >= end_date:
                    return existing

        # 获取日线行情
        try:
            daily = self.pro.daily(
                ts_code=ts_code,
                start_date=actual_start,
                end_date=end_date,
                fields="ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
            )
        except Exception as e:
            print(f"  [WARN] {ts_code} 日线获取失败: {e}")
            return pd.DataFrame()

        if daily.empty:
            return pd.read_csv(csv_path) if os.path.exists(csv_path) else daily

        # 统一 trade_date 为字符串，避免 merge 时 str/int 类型冲突
        daily["trade_date"] = daily["trade_date"].astype(str)

        # 获取换手率 + 基本面指标（需要 daily_basic 接口）
        try:
            basic = self.pro.daily_basic(
                ts_code=ts_code,
                start_date=actual_start,
                end_date=end_date,
                fields="ts_code,trade_date,turnover_rate,pe_ttm,pb"
            )
            if not basic.empty:
                basic["trade_date"] = basic["trade_date"].astype(str)
                daily = daily.merge(basic, on=["ts_code", "trade_date"], how="left")
        except Exception:
            daily["turnover_rate"] = 0.0

        # 确保 turnover_rate 列存在
        if "turnover_rate" not in daily.columns:
            daily["turnover_rate"] = 0.0

        daily = daily.sort_values("trade_date").reset_index(drop=True)

        # 写文件
        if append_mode and os.path.exists(csv_path):
            existing = pd.read_csv(csv_path, dtype={"trade_date": str})
            combined = pd.concat([existing, daily], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["ts_code", "trade_date"], keep="last"
            )
            combined = combined.sort_values("trade_date").reset_index(drop=True)
            combined.to_csv(csv_path, index=False, encoding="utf-8-sig")
            return combined
        else:
            daily.to_csv(csv_path, index=False, encoding="utf-8-sig")
            return daily

    # ------------------------------------------------------------------
    #  批量下载
    # ------------------------------------------------------------------
    def fetch_all(self, start_date: str = "20150101", end_date: str = None):
        """批量下载全市场（已过滤）股票日线数据。"""
        if end_date is None:
            end_date = time.strftime("%Y%m%d")

        codes = self.get_filtered_codes()

        total = len(codes)
        print(f"开始下载 {total} 只股票日线数据 ({start_date} ~ {end_date})")

        success, fail, skip = 0, 0, 0
        for i, code in enumerate(codes, 1):
            csv_path = os.path.join(self.daily_dir, f"{code}.csv")
            # 增量：已存在且最新的跳过
            if os.path.exists(csv_path):
                try:
                    existing = pd.read_csv(csv_path, dtype={"trade_date": str})
                    if not existing.empty and str(existing["trade_date"].max()) >= end_date:
                        skip += 1
                        continue
                except Exception:
                    pass

            try:
                df = self.fetch_daily(code, start_date, end_date)
                rows = len(df) if not df.empty else 0
                success += 1
                print(f"  [{i}/{total}] {code}: {rows} 行")
            except Exception as e:
                fail += 1
                print(f"  [{i}/{total}] {code}: 失败 - {e}")

            # Tushare 限频控制
            time.sleep(0.35)

        print(f"\n下载完成: 成功 {success}, 失败 {fail}, 跳过 {skip}")
        print(f"数据目录: {self.daily_dir}")


def main():
    parser = argparse.ArgumentParser(description="全A股（主板）数据下载")
    parser.add_argument("--token", default=None, help="Tushare Pro API Token（默认从 .env 读取）")
    parser.add_argument("--start", default="20150101", help="起始日期 (默认 20150101)")
    parser.add_argument("--end", default=None, help="结束日期 (默认今天)")
    parser.add_argument("--data-dir", default="data", help="数据存储目录 (默认 data)")
    args = parser.parse_args()

    token = args.token or os.getenv("TUSHARE_TOKEN", "")
    if not token:
        print("错误: 未提供 Tushare Token。请在 .env 中设置 TUSHARE_TOKEN 或使用 --token 参数")
        sys.exit(1)

    dl = TushareDownloader(token=token, data_dir=args.data_dir)
    dl.fetch_stock_basic()
    dl.fetch_index_daily(start_date=args.start, end_date=args.end)
    dl.fetch_all(start_date=args.start, end_date=args.end)


if __name__ == "__main__":
    main()
