AlphaGPT 仓库速读

这是一套 AI 驱动的沪深300量化选股系统。核心思路：用 Transformer 模型自动生成可解释的 Alpha 因子公式，通过截面回测打分筛选，输出每日 Top N 选股信号。

代码组织（按功能划分）
- run_daily.py：每日策略入口（数据更新 -> 训练 -> 信号输出，一步到位）
- data_download.py：数据采集。通过 Tushare Pro 拉取沪深300成分股及日线行情，存为 CSV。
- report.py：快捷训练 + 信号生成。
- times.py：独立实验脚本，单 ETF Alpha 挖矿（研究用，与主流程无关）。
- model_core/：核心模块，策略挖掘引擎。
  - config.py：A 股参数配置（佣金万2.5、印花税千1、涨跌停、T+1 等）。
  - data_loader.py：AshareDataLoader，从 CSV 构建特征张量 [num_stocks, 14, T]。
  - factors.py：FeatureEngineer，9 维因子计算与截面标准化。
  - ops.py：12 个数学算子（ADD, SUB, GATE, DECAY 等）。
  - vm.py：StackVM，栈式虚拟机，正序执行公式 token 序列。
  - alphagpt.py：核心 Transformer 模型（Looped Transformer + SwiGLU + QK-Norm + LoRD 正则化 + MTP Head）。
  - backtest.py：AshareBacktest，截面回测引擎（Sortino 评分 + 回撤惩罚 + 换手惩罚）。
  - engine.py：AlphaEngine，RL 训练循环（REINFORCE + Critic），含 Action Masking 和公式去重。
  - signal_writer.py：信号输出，CSV 格式（Top 30 + 全量排名）。
- data/：本地数据存储（constituents/hs300.csv + daily/*.csv）。
- signals/：输出信号目录。
- lord/：研究材料（LoRD 正则化实验）。
- paper/：学术论文。

主流程（从数据到信号）
1) data_download.py 拉取沪深300成分股及日线行情 -> data/ 目录
2) model_core/engine.py 训练生成最优公式（best_ashare_strategy.json）
3) 对最新交易日截面打分，输出 Top 30 选股信号至 signals/ 目录

运行方式
- 完整流程：python run_daily.py
- 只更新数据：python run_daily.py --update-only
- 跳过数据更新：python run_daily.py --skip-update
- 用已有公式生成信号：python run_daily.py --signal-only

核心思想
- 不是直接预测价格，而是"生成公式 -> 解释执行 -> 回测评分 -> 优化生成器"。
- 公式 = token 序列；token 由"特征 + 算子"组成，StackVM 正序执行成因子信号。
- 截面选股：每日对沪深300成分股打分排序，做多 Top N。

当前因子与算子一览
- 因子（FeatureEngineer，14 维）
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
- 佣金：万 2.5（双边）。印花税：千 1（卖出）。
- 涨跌停：主板 10% / 创业板科创板 20%。
- 换手率过滤：<0.5% 视为停牌/流动性不足，排除。
- 选股数量：Top N（截面排名前 10）。
- 训练步数：500 步，batch=1024，公式最大长度 15 tokens。

信号输出
- training_history.json：训练历史记录，最佳因子公式及得分。
- signals/{timestamp}/signals_top10.csv：最新交易日 Top N 选股。
- signals/{timestamp}/signals_all.csv：全部股票信号排名（含 direction 和 market_trend 列）。
- direction 基于截面中位数区分多空；market_trend 列（沪深300 均价/MA60 趋势），趋势向下时 direction=0（观望）。

times.py 与 model_core 的差异
- times.py：单 ETF 时序择时（做多/做空/空仓），5 维特征，10 个算子（含 DELTA5/TS_ZSCORE/TS_RANK 等 WorldQuant 风格时序算子），标准 Transformer + REINFORCE。
- model_core：沪深300截面选股（Top N），14 维特征，12 个算子（含 GATE/JUMP/DECAY 等逻辑算子），增强 Transformer（Looped + SwiGLU + QK-Norm + LoRD）。
- 两者公式解析方向不同：times.py 逆序（RPN），model_core 正序（StackVM）。

现状与依赖
- 需要 Tushare Pro API Token（需积分权限获取完整沪深300日线数据）。
- Python 3.10+（推荐 3.11），PyTorch。
- 无 Dockerfile / docker-compose，需手动搭建环境。

Takeaway
这不是一套"预测模型"，而是一个"自动写因子的系统"：它用 Transformer 生成公式，用回测奖励训练公式生成器，再把高分公式输出为选股信号。核心设计亮点是将"策略研究"（公式生成）和"信号执行"（截面选股 + 交易规则）清晰分层。
