# AlphaGPT
## 项目简介

AlphaGPT 是一套 **AI 驱动的量化选股系统**，核心思路是用 Transformer 模型自动生成可解释的 Alpha 因子公式，通过截面回测打分筛选，输出全市场主板 Top N 选股信号。

## 主流程（从数据到信号）

1. **数据采集** — `data_download.py` 通过 Tushare Pro 拉取全市场主板股票日线行情，自动过滤次新股/科创板/创业板/ST股
2. **模型训练** — `model_core` 用 REINFORCE 策略梯度训练 Transformer 生成最优因子公式
3. **信号输出** — 对最新交易日截面打分，输出 Top 30 选股信号至 `signals/` 目录

## 环境准备

### 安装依赖

```bash
pip3 install -r requirements.txt
```

Python 3.10+（推荐 3.11）。

### 外部服务

| 服务 | 用途 |
|------|------|
| Tushare Pro API Token | 获取 A 股行情数据（注册：https://tushare.pro） |

### 配置 .env

在项目根目录创建 `.env` 文件：

```env
# Tushare Pro API
TUSHARE_TOKEN=your_tushare_token

# 数据存储目录
DATA_DIR=data
```

## 运行方式

### 完整流程（推荐）

```bash
# 数据更新 → 训练 → 信号输出，一步到位
python run_daily.py
```

### 分步运行

| 命令 | 说明 |
|------|------|
| `python run_daily.py --update-only` | 只更新数据，不训练 |
| `python run_daily.py --skip-update` | 跳过数据更新，直接训练 + 信号 |
| `python run_daily.py --signal-only` | 用已有公式直接生成信号（跳过训练） |

### 信号输出

训练完成后会生成：

| 文件 | 说明 |
|------|------|
| `best_ashare_strategy.json` | 最佳因子公式及得分 |
| `training_history.json` | 训练历史记录 |
| `signals/{timestamp}/signals_top30.csv` | 最新交易日 Top 30 选股 |
| `signals/{timestamp}/signals_all.csv` | 全部股票信号排名 |

运行后终端会直接打印当日 Top 30 选股结果：

```
=======================================================
  最新交易日选股信号: 2026-04-04
=======================================================
   排名  股票代码        信号分值
  ----  ------------  ----------
     1  600519.SH      +0.5231  做多
     2  000858.SZ      +0.4892  做多
     ...
```

## 核心思想

- **不是直接预测价格**，而是"生成公式 → 解释执行 → 回测评分 → 优化生成器"
- 公式 = token 序列，token 由"特征 + 算子"组成，StackVM 执行成因子信号
- 截面选股：每日对全市场主板股票打分排序，做多 Top N

## 因子与算子

**输入因子（14 维）：** RET（对数收益率）、RET5（5 日动量）、VOL_CHG（成交量变化）、AMT_RAT（成交额占比）、TURN（换手率）、PRESSURE（K 线实体压力）、DEV（MA20 偏离度）、RSI（相对强弱）、TREND（MA60 趋势）、FOMO（成交额加速度）、VOL_CLUSTER（波动率聚集）、HL_RANGE（高低价振幅）、CLOSE_POS（收盘在区间位置）、REALIZED_VOL（已实现波动率）

**算子（12 个）：** ADD、SUB、MUL、DIV、NEG、ABS、SIGN、GATE、JUMP、DECAY、DELAY1、MAX3

## 核心配置参数

| 参数 | 值 | 说明 |
|------|----|------|
| 标的池 | 全市场主板（~3000只） | 自动过滤次新股/科创板/创业板/北交所/ST |
| 基准指数 | 中证全指 (000985.SH) | 回测对比基准 |
| 回测指标 | Sortino Ratio | 含回撤惩罚 + 换手惩罚 |
| 佣金 | 万 2.5 | 双边 |
| 印花税 | 千 0.5（2023-08-28后） | 卖出单边，此前千 1 |
| 涨跌停 | 主板 10% / ST 5% | 回测中限制 |
| 交易制度 | T+1 | 开盘价对开盘价收益 |
| 选股数量 | Top 20 | 截面排名前 20 |
| 再平衡 | 每 10 个交易日 | 非再平衡日沿用持仓 |
| 次新股过滤 | 上市 ≥ 250 个自然日 | 动态点对点过滤 |
| 训练步数 | 500 步 | REINFORCE 策略梯度 |
| 公式最大长度 | 8 tokens | |

## 代码组织

```
AlphaGPT/
├── run_daily.py           # 每日策略入口（数据更新 → 训练 → 信号输出）
├── data_download.py       # Tushare 数据下载器（全市场主板 + 基准指数）
├── model_core/            # 核心模块
│   ├── config.py          # A股参数配置（佣金、印花税、涨跌停、T+1、股票池过滤等）
│   ├── data_loader.py     # AshareDataLoader：从 CSV 构建特征张量 + 股票池过滤（板块/ST/次新股）
│   ├── factors.py         # 特征工程
│   ├── ops.py             # 数学算子（ADD, SUB, GATE, DECAY 等）
│   ├── vm.py              # 栈式虚拟机，执行公式 token 序列
│   ├── model.py           # 核心 Transformer 模型 + LoRD 正则化 + MTP Head
│   ├── backtest.py        # AshareBacktest：截面回测引擎（Sortino 评分）
│   ├── engine.py          # AlphaEngine：RL 训练循环
│   ├── report.py          # StrategyReport：OOS 业绩评估 + 可视化
│   └── signal_writer.py   # 信号输出（CSV 格式）
├── data/                  # 本地数据存储
│   ├── constituents/      # stock_basic.csv（全市场元数据）+ benchmark_index.csv
│   └── daily/             # 个股日线数据 ({code}.csv)
├── signals/               # 输出信号目录
├── ashare/                # 分钟频回测（独立流水线）
├── lord/                  # 研究：LoRD 正则化实验
├── paper/                 # 学术论文
└── assets/                # README 图片资源
```
