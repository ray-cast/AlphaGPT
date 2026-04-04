# times.py vs model_core 架构对比文档

## 一、总体定位

| 维度 | times.py | model_core |
|------|----------|------------|
| 定位 | 单标的（ETF/指数）因子挖掘原型 | A 股沪深 300 截面选股生产系统 |
| 代码结构 | 单文件 ~500 行 | 模块化 10 个文件 |
| 数据维度 | 1 只标的 × T 天 | N 只股票 × T 天（3D 张量） |
| 输出 | PNG 净值曲线 | JSON 策略 + CSV 信号 + PNG 报告 |

---

## 二、数据层

### 2.1 数据源

| | times.py | model_core |
|---|---|---|
| 数据来源 | Tushare API（单标的日线） | 本地 CSV 文件（沪深 300 成分股日线） |
| 数据类 | `DataEngine` | `AshareDataLoader` |
| 缓存方式 | Parquet 单文件 (`data_cache_final.parquet`) | CSV 目录结构 (`data/daily/{code}.csv`) |
| 标的范围 | 单只 ETF/指数，如 `511260.SH` | 最多 300 只成分股 |
| 日期对齐 | 不需要 | 取所有股票的公共交易日交集 |
| 时间范围 | `20150101 ~ 20250101` | 由 CSV 数据决定 |

### 2.2 目标收益率

两者均使用 **Open-to-Open** 收益率以符合 A 股 T+1 规则：

```
target_ret[t] = (open[t+2] - open[t+1]) / open[t+1]
```

- `times.py`：1D 张量 `[T]`
- `model_core`：2D 张量 `[num_stocks, T]`

### 2.3 训练/测试切分

两者均使用 **80 / 20** 时间切分，但粒度不同：

- `times.py`：`split_idx = int(len(df) * 0.8)` — 单序列前 80%
- `model_core`：`split_idx = int(len(dates) * 0.8)` — 所有股票共用同一时间切分点

---

## 三、因子体系

### 3.1 因子数量与列表

| # | times.py (5 维) | # | model_core (14 维) |
|---|---|---|---|
| 0 | `RET` — 日收益率 | 0 | `RET` — 日对数收益率 |
| 1 | `RET5` — 5 日累计收益 | 1 | `RET5` — 5 日累计收益 |
| 2 | `VOL_CHG` — 成交量变化率 vs MA20 | 2 | `VOL_CHG` — 成交量变化率 vs MA20 |
| 3 | `V_RET` — 成交量加权收益 | 3 | `AMT_RAT` — 成交额比 vs MA20 |
| 4 | `TREND` — 价格 vs MA60 | 4 | `TURN` — 换手率（截面标准化） |
| | | 5 | `PRESSURE` — 买卖压力（K 线实体/振幅） |
| | | 6 | `DEV` — 偏离 20 日均线 |
| | | 7 | `RSI` — 相对强弱指标 |
| | | 8 | `TREND` — 价格 vs MA60 |
| | | 9 | `FOMO` — 成交额加速度 |
| | | 10 | `VOL_CLUSTER` — 波动率聚集 |
| | | 11 | `HL_RANGE` — 高低价振幅 |
| | | 12 | `CLOSE_POS` — 收盘在区间位置 |
| | | 13 | `REALIZED_VOL` — 已实现波动率 |

### 3.2 标准化方式

| | times.py | model_core |
|---|---|---|
| 方法 | Robust Norm (MAD) | Robust Norm (MAD) |
| 维度 | 时序维度（全时间段） | **截面维度**（每天对所有股票独立标准化） |
| 实现 | NumPy `np.nanmedian` | PyTorch `torch.nanmedian(dim=1)` |

> **关键差异**：`model_core` 的标准化沿截面（dim=1，股票维度）进行，这对截面选股至关重要——确保每天不同股票之间的因子可比。

---

## 四、算子体系

### 4.1 算子列表

| | times.py (5 个算子) | | model_core (12 个算子) |
|---|---|---|---|
| `ADD` | x + y | `ADD` | x + y |
| `SUB` | x - y | `SUB` | x - y |
| `MUL` | x * y | `MUL` | x * y |
| `DIV` | x / (y + 1e-6) | `DIV` | x / (y + 1e-6) |
| `NEG` | -x | `NEG` | -x |
| `ABS` | \|x\| | `ABS` | \|x\| |
| `SIGN` | sign(x) | `SIGN` | sign(x) |
| `DELTA5` | x - delay(x, 5) | `GATE` | 条件门控 (3 元) |
| `MA20` | decay_linear(x, 20) | `JUMP` | 跳跃检测 (z-score > 3) |
| `STD20` | zscore(x, 20) | `DECAY` | x + 0.8·delay₁ + 0.6·delay₂ |
| `TS_RANK20` | rank(x, 20) | `DELAY1` | delay(x, 1) |
| | | `MAX3` | max(x, delay₁, delay₂) |

### 4.2 JIT 编译

- `times.py`：所有时序算子使用 `@torch.jit.script` 装饰
- `model_core`：同样使用 `@torch.jit.script`

### 4.3 新增算子语义

| 算子 | 含义 |
|------|------|
| `GATE(c, x, y)` | 条件门控：当 c > 0 时取 x，否则取 y（三元算子，arity=3） |
| `JUMP(x)` | 跳跃检测：对 x 做 z-score，只保留超过 3σ 的正向跳跃 |
| `DECAY(x)` | 指数衰减：x + 0.8·x[-1] + 0.6·x[-2]，近似指数加权 |
| `DELAY1(x)` | 简单延迟 1 天 |
| `MAX3(x)` | 近 3 天最大值 |

---

## 五、虚拟机（公式执行器）

| | times.py `solve_one()` | model_core `StackVM` |
|---|---|---|
| 解析方向 | **倒序**（Reverse Polish） | **正序**（标准 Prefix / Polish Notation） |
| 数据结构 | 手动栈操作 | 封装为独立 `StackVM` 类 |
| 输入形状 | `[T]` — 1D 因子向量 | `[num_stocks, T]` — 2D 因子矩阵 |
| NaN 处理 | `torch.nan_to_num(res)` | `torch.nan_to_num` + `posinf` / `neginf` 单独处理 |
| 常数因子过滤 | `std < 1e-4` → 返回 None | `std < 1e-4` → 返回 None |
| 批处理 | `solve_batch()` 逐个循环 | 通过**公式去重**优化（见训练部分） |

---

## 六、模型架构

### 6.1 整体对比

```
times.py AlphaGPT:
  TokenEmb → PosEmb → TransformerEncoder(2层) → LayerNorm → Actor-Head + Critic-Head

model_core AlphaGPT:
  TokenEmb → PosEmb → LoopedTransformer(2层×3循环) → RMSNorm → MTPHead(3任务) + Critic-Head
```

### 6.2 逐组件对比

| 组件 | times.py | model_core |
|------|----------|------------|
| **d_model** | 64 | 64 |
| **n_head** | 4 | 4 |
| **层数** | 2 | 2 |
| **Position Embedding** | 可学习参数 `[1, MAX_SEQ_LEN+1, 64]` | 可学习参数 `[1, MAX_FORMULA_LEN+1, 64]` |
| **Transformer** | `nn.TransformerEncoder`（标准） | `LoopedTransformer`（每层循环 3 次） |
| **归一化** | `nn.LayerNorm` | `RMSNorm` |
| **FFN** | 标准 Linear(64→128→64) | `SwiGLU`(64→256→64，含门控) |
| **注意力 QK** | 标准 | `QKNorm`（Query-Key 独立 L2 归一化 + 可学习缩放） |
| **Dropout** | 无 | 0.1 |
| **输出头** | `nn.Linear(64, vocab_size)` | `MTPHead`：3 个子头 + 路由网络加权 |
| **Critic** | `nn.Linear(64, 1)` | `nn.Linear(64, 1)` |

### 6.3 Looped Transformer 细节

`model_core` 的每层执行 **3 次循环迭代**，等效于把 2 层 Transformer 扩展为约 6 层的计算深度，但参数量不变：

```python
for _ in range(num_loops):  # 3 次
    x = x + Attention(RMSNorm(x))
    x = x + SwiGLU(RMSNorm(x))
```

### 6.4 MTPHead 多任务头

`model_core` 用路由网络动态选择 3 个任务头的加权组合：

```python
task_probs = softmax(MLP(last_emb))          # [B, 3]
task_outputs = [head_i(last_emb) for i in 3]  # [B, 3, vocab_size]
logits = (task_probs * task_outputs).sum(dim=1)
```

### 6.5 正则化

| | times.py | model_core |
|---|---|---|
| 权重衰减 | AdamW `weight_decay=1e-5` | AdamW 无显式 weight_decay |
| LoRD 正则 | 无 | **Newton-Schulz Low-Rank Decay** — 对 Q/K 投影矩阵施加低秩约束 |
| Stable Rank 监控 | 无 | 每 100 步记录注意力矩阵有效秩 |
| Dropout | 无 | 0.1 |

#### LoRD 原理

使用 Newton-Schulz 迭代逼近最大奇异向量方向，沿该方向施加衰减，使权重矩阵趋向低秩：

```
Y_{k+1} = 0.5 * Y_k * (3I - Y_k^T * Y_k)   # 收敛到正交矩阵
W ← W - decay_rate * Y                        # 沿主奇异方向衰减
```

---

## 七、训练流程

### 7.1 超参数

| 超参数 | times.py | model_core |
|--------|----------|------------|
| Batch Size | 1024 | 1024 |
| 训练步数 | 400 | 500 |
| 公式最大长度 | 8 | 10 |
| 学习率 | `3e-4` | `1e-3` |
| 优化器 | AdamW (`weight_decay=1e-5`) | AdamW (`lr=1e-3`) + LoRD |
| RL 算法 | REINFORCE + Critic | REINFORCE + Critic |

### 7.2 Action Masking

两者的 Action Masking 逻辑相同，确保生成合法前缀表达式树：

- `open_slots == 0` → 填充第一个 feature
- `open_slots >= remaining_steps` → 强制选 feature
- 否则 feature 和 op 均可选

### 7.3 公式去重（model_core 独有）

`model_core` 在训练循环中引入了**公式去重**优化：

```
1. 计算每个公式的合法前缀长度 → trimmed
2. 以 tuple(trimmed) 为 key 查重
3. 相同公式只执行一次 VM + 回测
4. 结果映射回 batch 中所有相同公式的位置
```

这在 BATCH_SIZE=1024 时可显著减少重复计算。

### 7.4 优势函数

| | times.py | model_core |
|---|---|---|
| 优势 | `adv = rewards - baseline` | `adv = (rewards - baseline) / (rewards.std() + 1e-5)` |

`model_core` 对优势做了**标准化**，使梯度更新更稳定。

### 7.5 Loss 计算

| | times.py | model_core |
|---|---|---|
| Policy Loss | `-(log_probs.sum(1) * adv).mean()` | 逐步累加 `-(log_probs[t] * adv).mean()` |
| Value Loss | `0.5 * MSE(value_pred, rewards)` | `0.5 * MSE(value_pred, rewards)` |
| 总 Loss | `policy_loss + 0.5 * value_loss` | `policy_loss + 0.5 * value_loss` |

> 两者 Loss 公式等价，`model_core` 的逐步累加写法与 `times.py` 的批量乘法写法在数学上相同。

### 7.6 训练历史记录

| | times.py | model_core |
|---|---|---|
| 记录方式 | tqdm 实时显示 | JSON 文件 (`training_history.json`) |
| 记录内容 | Valid 比例、Best Sortino | AvgReward、BestScore、Unique 比例、StableRank |
| 最佳策略保存 | 内存中 | JSON 文件 (`best_ashare_strategy.json`) |

---

## 八、回测系统

### 8.1 训练时回测（评估奖励）

| | times.py `backtest()` | model_core `AshareBacktest.evaluate()` |
|---|---|---|
| **回测模式** | 单标的时序 | A 股截面选股 |
| **输入形状** | `[B, T]`（batch 个因子，每个长度 T） | `[num_stocks, T]`（全股票截面因子） |
| **信号生成** | `sign(tanh(f))` → 多/空/空仓三态 | 截面排序 → 做多 Top-30 |
| **持仓方式** | 时序多空（单标的，每天可做多或做空） | 截面纯多头（每天选 30 只股票等权做多） |
| **成本模型** | 统一 `COST_RATE = 0.0005`（万五双边） | 拆分：买入万 2.5 + 卖出万 7.5（含印花税千 1） |
| **过滤机制** | 无 | 换手率 < 0.5% 的股票排除（停牌/流动性不足） |
| **Fitness 指标** | Sortino 比率 | Sortino 比率 - 回撤惩罚 - 换手惩罚 |
| **回撤惩罚** | 无 | 回撤超 5% 额外惩罚 |
| **奖励范围** | `clamp(-3, 5)` | 无 clamp（仅 Sortino 有 -3~5 clamp） |
| **数据使用** | 仅训练集前 80% | 仅训练集前 80% |

#### times.py 时序回测流程

```
factor → tanh → sign → position (多/空/空)
pnl = position * target_ret - turnover * cost
reward = Sortino(pnl) 年化
```

#### model_core 截面回测流程

```
factor [num_stocks, T] → 每天：
  1. 排除 turnover_rate < 0.5% 的股票
  2. 按因子值排序，选 Top-30
  3. position[top_30, t] = 1.0（等权多头）
daily_pnl = sum(position * target_ret - turnover * cost) / 30
fitness = Sortino(daily_pnl) - dd_penalty - turnover_penalty
```

### 8.2 样本外评估

| | times.py `final_reality_check()` | model_core `StrategyReport` |
|---|---|---|
| 类/函数 | 函数 | 独立类 `StrategyReport` |
| 输出指标 | 年化收益、波动率、夏普、最大回撤、Calmar | 同左 + 胜率、换手率、基准对比、超额收益 |
| 净值曲线 | `strategy_performance.png` | `reports/oos_performance_{run_id}.png` |
| 基准 | 同标的 Buy & Hold | 沪深 300 等权均值 |
| 信号平滑 | 3 日滚动平均 | 无 |
| 报告格式 | `print()` | `print_report()` + PNG 图 |

---

## 九、信号输出

| | times.py | model_core |
|---|---|---|
| 信号输出 | 无（仅训练 + OOS 评估） | `SignalWriter` 输出 CSV 信号文件 |
| 输出内容 | — | 每只股票的 signal_score、rank、direction、market_trend |
| 文件格式 | — | `signals_all.csv`（全量）+ `signals_top30.csv`（精简版） |
| 市场趋势过滤 | 无 | MA60 趋势下行时 direction=0（观望） |

---

## 十、文件结构对比

### times.py（单文件）

```
times.py
├── 配置常量 (INDEX_CODE, BATCH_SIZE, ...)
├── 时序算子 (_ts_delay, _ts_delta, _ts_zscore, ...)
├── AlphaGPT (模型)
├── DataEngine (数据加载)
├── DeepQuantMiner (训练 + 回测 + VM)
└── final_reality_check (OOS 评估)
```

### model_core（模块化）

```
model_core/
├── config.py         — 全局配置 (ModelConfig)
├── factors.py        — 14 维因子计算 (FeatureEngineer)
├── ops.py            — 12 个算子定义
├── vm.py             — 栈式虚拟机 (StackVM)
├── alphagpt.py       — 模型定义 (AlphaGPT + LoRD + LoopedTransformer)
├── data_loader.py    — A 股数据加载 (AshareDataLoader)
├── backtest.py       — 截面回测 + 单标的回测
├── engine.py         — 训练引擎 (AlphaEngine)
├── signal_writer.py  — 信号输出 (SignalWriter)
└── report.py         — OOS 报告 (StrategyReport)
```

---

## 十一、升级总结

从 `times.py` 到 `model_core` 的核心演进方向：

1. **从单标的到截面选股**：回测从单标的时序多空升级为沪深 300 截面 Top-N 选股
2. **从 5 维到 14 维因子**：引入换手率、买卖压力、RSI、波动率聚集等新因子
3. **从简单算子到丰富算子**：新增 GATE（条件门控）、JUMP（跳跃检测）等高级算子
4. **模型架构现代化**：Looped Transformer + QK-Norm + SwiGLU + RMSNorm + MTPHead
5. **正则化增强**：Newton-Schulz LoRD 低秩约束 + Dropout
6. **训练效率优化**：公式去重、优势标准化、更高学习率
7. **工程完整度**：模块化拆分、JSON 持久化、CSV 信号输出、独立报告模块
