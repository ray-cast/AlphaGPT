import torch
import os

# 项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ModelConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- 数据路径 ----------
    DATA_DIR = os.path.join(_PROJECT_ROOT, "data")

    # ---------- 数据集日期划分 ----------
    DATA_START_DATE = "20190101"         # 数据起始日期
    TRAIN_END_DATE = "20241231"          # 训练集截止（含）
    TEST_END_DATE = "20250630"           # 测试集截止（含）
    # 验证集：TEST_END_DATE 之后的所有数据

    # ---------- 训练参数 ----------
    BATCH_SIZE = 1024
    TRAIN_STEPS = 500
    MAX_FORMULA_LEN = 15

    # ---------- A股交易成本 ----------
    COMMISSION_RATE = 0.00025      # 佣金万2.5（双向）
    STAMP_DUTY_RATE = 0.0005       # 印花税千1（仅卖出）
    TOTAL_BUY_COST = 0.00025       # 买入成本
    TOTAL_SELL_COST = 0.00025 + 0.0005   # 卖出成本（佣金+印花税）

    # ---------- A股交易规则 ----------
    PRICE_LIMIT_MAIN = 0.10        # 主板涨跌停 ±10%
    PRICE_LIMIT_GEM = 0.20         # 创业板/科创板 ±20%
    MIN_LOT_SIZE = 100             # 最小交易单位（股）
    MIN_TURNOVER_RATE = 0.005      # 最低换手率（过滤停牌/流动性不足）

    # ---------- 早停 ----------
    PATIENCE_LIMIT = 50           # 连续 N 步无新 best 则早停

    # ---------- 探索与多样性 ----------
    ENTROPY_COEF_START = 0.20      # 起始 entropy 系数
    ENTROPY_COEF_END = 0.02        # 终止 entropy 系数
    WARMUP_STEPS = 20              # 前 N 步强制均匀采样
    LENGTH_BONUS_COEF = 0.1        # 每 log2(公式长度) 的 advantage 加成
    DIVERSITY_TARGET = 0.3         # 低于此 unique ratio 时启用多样性惩罚
    DIVERSITY_PENALTY = 0.5        # 重复公式的 advantage 惩罚强度
    NOVELTY_BONUS = 0.2            # 首次出现公式的额外奖励
    MIXED_STRUCTURE_BONUS = 0.3    # 时序+截面混合公式的额外奖励

    # ---------- 因子维度 ----------
    INPUT_DIM = 14

    # ---------- 信号输出 ----------
    SIGNAL_DIR = os.path.join(_PROJECT_ROOT, "signals")
    SIGNAL_THRESHOLD = 0.7         # sigmoid 阈值
    TOP_N_STOCKS = 10              # 截面选股数量
