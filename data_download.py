"""
沪深300成分股数据下载脚本

用法:
    python data_download.py [--start 20150101] [--end 20260404]

    Token 优先从 .env 文件读取 TUSHARE_TOKEN，也可通过 --token 参数覆盖。

数据目录结构:
    data/
      constituents/
        hs300.csv              # 成分股列表
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
from dotenv import load_dotenv

try:
    import tushare as ts
except ImportError:
    print("请先安装 tushare: pip install tushare")
    sys.exit(1)

# 加载 .env 文件中的环境变量
# override=True 确保 .env 文件中的设置覆盖系统环境变量
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
    #  成分股列表
    # ------------------------------------------------------------------
    def fetch_constituents(self) -> pd.DataFrame:
        """获取沪深300最新成分股列表，缓存到 CSV。"""
        cache_path = os.path.join(self.const_dir, "hs300.csv")

        # 尝试获取最近一期的成分股权重
        try:
            # 用最近日期获取成分股
            trade_cal = self.pro.trade_cal(
                exchange="SSE", is_open=1,
                start_date="20250101", end_date="20260404",
                fields="cal_date"
            )
            latest_date = trade_cal["cal_date"].iloc[-1]
            df = self.pro.index_weight(
                index_code="000300.SH",
                start_date=latest_date,
                fields="index_code,con_code,weight,trade_date"
            )
            if df.empty:
                # 回退到更早日期
                df = self.pro.index_weight(
                    index_code="000300.SH",
                    start_date="20250101",
                    end_date="20260404",
                    fields="index_code,con_code,weight,trade_date"
                )
                # 取最后一期
                last_date = df["trade_date"].max()
                df = df[df["trade_date"] == last_date]
        except Exception:
            # 兜底：用 hs300 接口
            df = self.pro.hs300(
                fields="ts_code,name,weight",
                start_date="20250101",
                end_date="20260404"
            )
            df = df.rename(columns={"ts_code": "con_code"})
            df["index_code"] = "000300.SH"

        df.to_csv(cache_path, index=False, encoding="utf-8-sig")
        print(f"成分股列表已保存: {cache_path} ({len(df)} 只)")
        return df

    def load_constituent_codes(self) -> list[str]:
        """从缓存读取成分股代码列表。"""
        cache_path = os.path.join(self.const_dir, "hs300.csv")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"未找到成分股缓存 {cache_path}，请先运行 fetch_constituents()"
            )
        df = pd.read_csv(cache_path)
        col = "con_code" if "con_code" in df.columns else "ts_code"
        codes = df[col].dropna().unique().tolist()
        print(f"从缓存加载 {len(codes)} 只成分股")
        return codes

    # ------------------------------------------------------------------
    #  历史成分股（消除幸存者偏差）
    # ------------------------------------------------------------------
    def fetch_constituents_history(
        self, start_date: str = "20160101", end_date: str = None
    ) -> pd.DataFrame:
        """获取沪深300全部历史成分股权重，缓存到 CSV。

        按年分段查询 index_weight 接口，避免 Tushare 单次 6000 行限制导致只拿到最新数据。
        消除幸存者偏差的关键数据源。
        """
        import datetime as _dt
        if end_date is None:
            end_date = time.strftime("%Y%m%d")
        cache_path = os.path.join(self.const_dir, "hs300_history.csv")

        start_y = int(start_date[:4])
        end_y = int(end_date[:4])

        # 按年分段查询（每年 ~3600 行，远低于 6000 行限制）
        all_pages = []
        for y in range(start_y, end_y + 1):
            y_start = max(f"{y}0101", start_date)
            y_end = min(f"{y}1231", end_date)
            try:
                df = self.pro.index_weight(
                    index_code="000300.SH",
                    start_date=y_start,
                    end_date=y_end,
                    fields="index_code,con_code,weight,trade_date",
                )
            except Exception as e:
                print(f"  [WARN] 历史成分股查询失败 ({y_start}~{y_end}): {e}")
                continue

            if df is not None and not df.empty:
                all_pages.append(df)
                n_dates = df["trade_date"].nunique()
                print(f"  {y}: {len(df)} 行, {n_dates} 个调整期")
            time.sleep(0.35)

        if not all_pages:
            print("  [WARN] 未获取到历史成分股数据，使用缓存（如有）")
            if os.path.exists(cache_path):
                return pd.read_csv(cache_path)
            return pd.DataFrame()

        result = pd.concat(all_pages, ignore_index=True)
        result = result.drop_duplicates(subset=["con_code", "trade_date"], keep="last")

        # 与已有缓存合并
        if os.path.exists(cache_path):
            existing = pd.read_csv(cache_path)
            result = pd.concat([existing, result], ignore_index=True)
            result = result.drop_duplicates(subset=["con_code", "trade_date"], keep="last")

        result = result.sort_values(["trade_date", "con_code"]).reset_index(drop=True)
        result.to_csv(cache_path, index=False, encoding="utf-8-sig")

        rebalance_dates = result["trade_date"].nunique()
        unique_codes = result["con_code"].nunique()
        date_range = f"{result['trade_date'].min()}~{result['trade_date'].max()}"
        print(f"历史成分股已保存: {cache_path} "
              f"({rebalance_dates} 个调整期, {unique_codes} 只股票, {date_range})")
        return result

    # ------------------------------------------------------------------
    #  HS300 指数日线（正确基准）
    # ------------------------------------------------------------------
    def fetch_index_daily(self, start_date: str = "20160101", end_date: str = None):
        """下载沪深300指数日线行情，作为回测基准。"""
        if end_date is None:
            end_date = time.strftime("%Y%m%d")
        cache_path = os.path.join(self.const_dir, "hs300_index.csv")

        actual_start = start_date
        append_mode = False
        if os.path.exists(cache_path):
            existing = pd.read_csv(cache_path)
            if not existing.empty:
                last_date = str(existing["trade_date"].max())
                actual_start = last_date
                append_mode = True
                if last_date >= end_date:
                    print(f"HS300指数数据已是最新 ({last_date})")
                    return existing

        try:
            df = self.pro.index_daily(
                ts_code="000300.SH",
                start_date=actual_start,
                end_date=end_date,
            )
        except Exception as e:
            print(f"  [WARN] HS300指数获取失败: {e}")
            if os.path.exists(cache_path):
                return pd.read_csv(cache_path)
            return pd.DataFrame()

        if df.empty:
            return pd.read_csv(cache_path) if os.path.exists(cache_path) else df

        df = df.sort_values("trade_date").reset_index(drop=True)

        if append_mode and os.path.exists(cache_path):
            existing = pd.read_csv(cache_path)
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["trade_date"], keep="last")
            combined = combined.sort_values("trade_date").reset_index(drop=True)
            combined.to_csv(cache_path, index=False, encoding="utf-8-sig")
            print(f"HS300指数已更新: {cache_path} ({len(combined)} 行)")
            return combined
        else:
            df.to_csv(cache_path, index=False, encoding="utf-8-sig")
            print(f"HS300指数已保存: {cache_path} ({len(df)} 行)")
            return df

    def load_history_constituent_codes(self) -> list[str]:
        """从缓存读取全部历史成分股代码（并集），用于扩展下载范围。"""
        cache_path = os.path.join(self.const_dir, "hs300_history.csv")
        if not os.path.exists(cache_path):
            return []
        df = pd.read_csv(cache_path)
        col = "con_code" if "con_code" in df.columns else "ts_code"
        codes = df[col].dropna().unique().tolist()
        print(f"从历史缓存加载 {len(codes)} 只成分股（全部时期并集）")
        return codes

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
            existing = pd.read_csv(csv_path)
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

        # 获取换手率（需要 daily_basic 接口）
        try:
            basic = self.pro.daily_basic(
                ts_code=ts_code,
                start_date=actual_start,
                end_date=end_date,
                fields="ts_code,trade_date,turnover_rate"
            )
            if not basic.empty:
                daily = daily.merge(basic, on=["ts_code", "trade_date"], how="left")
        except Exception:
            daily["turnover_rate"] = 0.0

        # 确保 turnover_rate 列存在
        if "turnover_rate" not in daily.columns:
            daily["turnover_rate"] = 0.0

        daily = daily.sort_values("trade_date").reset_index(drop=True)

        # 写文件
        if append_mode and os.path.exists(csv_path):
            existing = pd.read_csv(csv_path)
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
        """批量下载所有沪深300成分股日线数据。"""
        if end_date is None:
            end_date = time.strftime("%Y%m%d")

        # 确保有成分股列表（当前 + 历史，消除幸存者偏差）
        try:
            codes_current = self.load_constituent_codes()
        except FileNotFoundError:
            print("成分股缓存不存在，正在获取...")
            self.fetch_constituents()
            codes_current = self.load_constituent_codes()
        codes_history = self.load_history_constituent_codes()
        codes = sorted(set(codes_current) | set(codes_history))

        total = len(codes)
        print(f"开始下载 {total} 只股票日线数据 ({start_date} ~ {end_date})")

        success, fail = 0, 0
        for i, code in enumerate(codes, 1):
            try:
                df = self.fetch_daily(code, start_date, end_date)
                rows = len(df) if not df.empty else 0
                success += 1
                if i % 20 == 0 or i == total:
                    print(f"  [{i}/{total}] {code}: {rows} 行")
            except Exception as e:
                fail += 1
                print(f"  [{i}/{total}] {code}: 失败 - {e}")

            # Tushare 限频控制
            time.sleep(0.35)

        print(f"\n下载完成: 成功 {success}, 失败 {fail}")
        print(f"数据目录: {self.daily_dir}")


def main():
    parser = argparse.ArgumentParser(description="沪深300成分股数据下载")
    parser.add_argument("--token", default=None, help="Tushare Pro API Token（默认从 .env 读取）")
    parser.add_argument("--start", default="20150101", help="起始日期 (默认 20150101)")
    parser.add_argument("--end", default=None, help="结束日期 (默认今天)")
    parser.add_argument("--data-dir", default="data", help="数据存储目录 (默认 data)")
    parser.add_argument("--history-only", action="store_true",
                        help="仅下载历史成分股（不更新日线等数据）")
    args = parser.parse_args()

    token = args.token or os.getenv("TUSHARE_TOKEN", "")
    if not token:
        print("错误: 未提供 Tushare Token。请在 .env 中设置 TUSHARE_TOKEN 或使用 --token 参数")
        sys.exit(1)

    dl = TushareDownloader(token=token, data_dir=args.data_dir)

    if args.history_only:
        dl.fetch_constituents_history(start_date=args.start, end_date=args.end)
    else:
        dl.fetch_constituents()
        dl.fetch_constituents_history(start_date=args.start, end_date=args.end)
        dl.fetch_index_daily(start_date=args.start, end_date=args.end)
        dl.fetch_all(start_date=args.start, end_date=args.end)


if __name__ == "__main__":
    main()
