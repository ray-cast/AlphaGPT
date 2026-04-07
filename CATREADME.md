AlphaGPT 仓库速读

这是一套 AI 驱动的全市场量化选股系统。核心思路：用 Transformer 模型自动生成可解释的 Alpha 因子公式，通过截面回测打分筛选，输出每日 Top N 选股信号。

股票池与过滤
- 全市场主板（~3000只），自动过滤：科创板(688)、创业板(300/301)、北交所(8xx)、ST/*ST、次新股（上市<250天）。
- 次新股过滤为动态点对点：每个交易日根据 IPO 日期判断，上市满 250 自然日后才纳入。
- 基准指数：中证全指 (000985.SH)，替代原沪深300基准。

代码组织（按功能划分）
- run_daily.py：每日策略入口（数据更新 -> 训练 -> 信号输出，一步到位）。
- data_download.py：数据采集。通过 Tushare Pro 拉取全市场主板股票日线行情 + 基准指数，存为 CSV。
- model_core/：核心模块，策略挖掘引擎。
  - config.py：A 股参数配置（佣金万2.5、印花税分段、涨跌停、T+1、股票池过滤等）。
  - data_loader.py：AshareDataLoader，从 CSV 构建特征张量 [num_stocks, 20, T]，含股票池过滤（板块/ST/次新股）与次新股掩码。
  - factors.py：FeatureEngineer，特征因子计算与截面标准化。
  - ops.py：数学算子（ADD, SUB, GATE, DECAY 等）。
  - vm.py：PrefixVM，栈式虚拟机，正序执行公式 token 序列。
  - model.py：核心 Transformer 模型（Looped Transformer + SwiGLU + QK-Norm + LoRD 正则化 + MTP Head）。
  - backtest.py：AshareBacktest，截面回测引擎（Sortino 评分 + 回撤惩罚 + 换手惩罚）。
  - engine.py：AlphaEngine，RL 训练循环（REINFORCE + Critic），含 Action Masking 和公式去重。
  - report.py：StrategyReport，OOS 业绩评估 + 可视化。
  - signal_writer.py：信号输出，CSV 格式（Top 30 + 全量排名）。
- ashare/：分钟频回测（独立流水线，含 market/cn_rules.py A股交易规则）。
- data/：本地数据存储（constituents/stock_basic.csv + benchmark_index.csv + daily/*.csv）。
- signals/：输出信号目录。
- lord/：研究材料（LoRD 正则化实验）。
- paper/：学术论文。

主流程（从数据到信号）
1) data_download.py 拉取全市场主板股票日线行情 + 基准指数 -> data/ 目录
2) model_core/engine.py 训练生成最优公式（best_ashare_strategy.json / training_history.json）
3) 对最新交易日截面打分，输出 Top 30 选股信号至 signals/ 目录

运行方式
- 完整流程：python run_daily.py
- 只更新数据：python run_daily.py --update-only
- 跳过数据更新：python run_daily.py --skip-update
- 用已有公式生成信号：python run_daily.py --signal-only

核心思想
- 不是直接预测价格，而是"生成公式 -> 解释执行 -> 回测评分 -> 优化生成器"。
- 公式 = token 序列；token 由"特征 + 算子"组成，PrefixVM 正序执行成因子信号。
- 截面选股：每日对全市场主板股票打分排序，做多 Top N。

当前因子与算子一览
- 因子（FeatureEngineer，20 维）
  - RET：日对数收益率
  - RET5：5 日累计收益率
  - VOL_CHG：成交量变化率 vs 20 日均值
  - AMT_RAT：成交额占比 vs 20 日均值
  - TURN：换手率（截面标准化）
  - PRESSURE：K 线实体压力（body/range * tanh 压缩）
  - DEV：偏离 20 日均线
  - RSI：相对强弱（14 日 RSI 归一化到 [-1,1]）
  - TREND：价格 vs 60 日均线
  - FOMO：成交额加速度（amt_ratio 的一阶差分，捕捉资金流入爆发）
  - VOL_CLUSTER：波动率聚集（5 日方差 / 20 日方差，GARCH 效应）
  - HL_RANGE：高低价振幅（(high - low) / close，日内分歧度）
  - CLOSE_POS：收盘在区间位置（(close - low) / (high - low)，尾盘行为）
  - REALIZED_VOL：已实现波动率（sqrt(20 日滚动方差)，波动率绝对水平）
- 算子（OPS_CONFIG，12 个）
  - ADD：加法（二元）
  - SUB：减法（二元）
  - MUL：乘法（二元）
  - DIV：除法（二元，保护除数）
  - NEG：取负（一元）
  - ABS：绝对值（一元）
  - SIGN：符号（一元）
  - GATE：门控选择（三元，condition>0 选 x 否则 y）
  - JUMP：极端跳变检测（zscore>3 的部分保留）
  - DECAY：衰减叠加（t + 0.8*lag1 + 0.6*lag2）
  - DELAY1：滞后 1 天
  - MAX3：当前 / lag1 / lag2 最大值

训练机制
- 模型：Looped Transformer（2 层 x 3 循环 = 等效 6 层），d_model=64, n_head=4。
- 正则化：LoRD（Newton-Schulz 低秩衰减），监控 StableRank。
- 输出头：MTP Head（3 任务头 + 路由网络），多头选举生成 logits。
- 训练算法：REINFORCE + Critic baseline（Advantage 标准化）。
- Action Masking：通过 open_slots 追踪算子栈深度，确保每条采样公式都是合法的前缀表达式树，消除无效探索。
- 公式去重：batch 内相同 token 序列只执行一次 VM + Backtest，训练后期可减少 2-5x 执行量。
- 评分函数：Sortino ratio（下行标准差）+ 回撤惩罚（>5% 扣分）+ 换手惩罚。
- 超额收益：reward = 持仓收益 - 市场均值，让模型学 alpha 而非 beta。

回测与交易规则
- 收益计算：open-to-open（T+1 合规）。
- 佣金：万 2.5（双边）。印花税：千 0.5（2023-08-28 后卖出单边），此前千 1。
- 涨跌停：主板 10% / ST 5%（自动检测疑似 ST 的 5% 涨跌停行为）。
- 换手率过滤：<0.5% 视为停牌/流动性不足，排除。
- 选股数量：Top 20（截面排名前 20）。
- 再平衡：每 10 个交易日，非再平衡日沿用持仓；换仓排名阈值 5 名。
- 训练步数：500 步，batch=1024，公式最大长度 8 tokens。

信号输出
- training_history.json：训练历史记录，最佳因子公式及得分。
- signals/{timestamp}/signals_top30.csv：最新交易日 Top N 选股。
- signals/{timestamp}/signals_all.csv：全部股票信号排名（含 direction 和 market_trend 列）。

现状与依赖
- 需要 Tushare Pro API Token（需积分权限获取日线数据）。
- Python 3.10+（推荐 3.11），PyTorch。
- 无 Dockerfile / docker-compose，需手动搭建环境。

Takeaway
这不是一套"预测模型"，而是一个"自动写因子的系统"：它用 Transformer 生成公式，用回测奖励训练公式生成器，再把高分公式输出为选股信号。核心设计亮点是将"策略研究"（公式生成）和"信号执行"（截面选股 + 交易规则）清晰分层。
