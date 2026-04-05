import torch
import os

# 项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ModelConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- 数据路径 ----------
    DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
    BENCHMARK_INDEX = "000300.SH"   # 沪深300指数代码（回测基准）

    # ---------- 数据集日期划分 ----------
    DATA_START_DATE = "20170101"         # 数据起始日期
    VALID_END_DATE = "20181231"          # 验证集截止（含）— 熊市压力测试
    TRAIN_END_DATE = "20231231"          # 训练集截止（含）
    TEST_END_DATE = "20261231"           # 测试集截止（含）
    # 验证集：2017-2018（熊市） | 训练集：2019-2023 | 测试集：2024-2026

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
    DIVERSITY_TARGET = 0.3         # 低于此 unique ratio 时启用多样性惩罚
    DIVERSITY_PENALTY = 0.5        # 重复公式的 advantage 惩罚强度

    # ---------- 因子维度 ----------
    INPUT_DIM = 15

    # ---------- 信号输出 ----------
    SIGNAL_DIR = os.path.join(_PROJECT_ROOT, "signals")
    SIGNAL_THRESHOLD = 0.7         # sigmoid 阈值
    TOP_N_STOCKS = 10              # 截面选股数量
    REBALANCE_RANK_GAP = 5         # 换仓排名阈值：新候选领先最弱持仓至少N名才调仓（0=关闭）

    # ---------- 基本面过滤 ----------
    MAX_PE_TTM = 25                # 最高市盈率TTM（排除高估值；PE<=0 隐含排除亏损股即 EPS<=0）
    MIN_ROE = 0                    # 最低ROE（通过 PB/PE_TTM 近似；排除低效公司）

    # ---------- SFT 热启动 ----------
    SEED_FORMULA_NAMES = [         # 种子公式（名称格式，token ID 动态计算）
        ("CS_RANK", ["P_VALUE"]),                   # 价值因子截面排名
        ("CS_RANK", ["RET"]),                       # 动量截面排名
        ("CROSS", ["P_VALUE", "RET"]),              # 价值×动量交互
    ]
    SFT_STEPS = 50                 # SFT 预训练步数（0 = 关闭热启动）
