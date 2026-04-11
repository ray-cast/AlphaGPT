import torch
import os

# 项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ModelConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- 数据路径 ----------
    DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
    BENCHMARK_INDEX = "000985.SH"   # 中证全指（回测基准）

    # ---------- 数据集日期划分（半开区间 [start, end)） ----------
    VALID_START_DATE = "20170101"        # 验证集起始
    VALID_END_DATE = "20181231"          # 验证集截止（含）
    TRAIN_START_DATE = "20190101"        # 训练集起始
    TRAIN_END_DATE = "20231231"          # 训练集截止（含）
    TEST_START_DATE = "20240101"         # 测试集起始
    TEST_END_DATE = "20261231"           # 测试集截止（含）
    # 验证集：2017-2018（熊市） | 训练集：2019-2023 | 测试集：2024-2026

    # ---------- 训练参数 ----------
    BATCH_SIZE = 1024
    TRAIN_STEPS = 500
    MAX_FORMULA_LEN = 10

    # ---------- A股交易成本 ----------
    COMMISSION_RATE = 0.00025           # 佣金 万2.5（买卖双边）

    # ---------- A股交易规则 ----------
    MIN_LOT_SIZE = 100             # 最小交易单位（股）
    MIN_TURNOVER_RATE = 0.005      # 最低换手率（过滤停牌/流动性不足）

    # ---------- 早停 ----------
    PATIENCE_LIMIT = 50           # 连续 N 步无新 best 则早停
    MIN_TRAIN_STEPS = 300         # 至少跑 N 步再允许早停（等 entropy 退火到 ≈0.06）

    # ---------- PPO ----------
    PPO_EPOCHS = 4           # 每次 rollout 的 PPO 更新轮数
    PPO_CLIP_EPS = 0.2       # PPO clip epsilon
    GRAD_CLIP_NORM = 0.5     # 梯度裁剪范数

    # ---------- GAE (Generalized Advantage Estimation) ----------
    GAMMA = 0.99             # 折扣因子，权衡未来奖励
    GAE_LAMBDA = 0.95        # GAE参数，平衡偏差和方差（0=当前，1=累积）

    # ---------- 探索与多样性 ----------
    ENTROPY_COEF_START = 0.1        # 起始 entropy 系数 (提高探索)
    ENTROPY_COEF_END = 0.02         # 终止 entropy 系数 (保持适度探索)
    EPS_GREEDY_START = 0.3         # epsilon-greedy 起始值 (提高探索)
    EPS_GREEDY_END = 0.05          # epsilon-greedy 终止值 (保持适度探索)

    # ---------- IC 奖励 ----------
    IC_WEIGHT = 5.0              # IC 奖励权重（IC ≈ [-0.1, 0.1]，乘此系数与 Sortino 量级对齐）

    # ---------- 因子维度 ----------
    INPUT_DIM = 6

    # ---------- 信号输出 ----------
    SIGNAL_DIR = os.path.join(_PROJECT_ROOT, "signals")
    SIGNAL_THRESHOLD = 0.7         # sigmoid 阈值
    TOP_N_STOCKS = 20              # 截面选股数量

    # ---------- 全市场股票池 ----------
    UNIVERSE = "all"                # "hs300" 仅沪深300 | "all" 全A股主板
    EXCLUDE_GEM = True              # 排除创业板 (300xxx, 301xxx)
    EXCLUDE_STAR = True             # 排除科创板 (688xxx)
    EXCLUDE_BSE = True              # 排除北交所 (8xxxxx)
    EXCLUDE_ST = True               # 排除 ST/*ST 股票
    IPO_MIN_DAYS = 250              # 上市最少N个自然日（过滤次新股，≈1年）
