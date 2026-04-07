import torch
import os

# 项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ModelConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- 数据路径 ----------
    DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
    BENCHMARK_INDEX = "000985.SH"   # 中证全指（回测基准）

    # ---------- 数据集日期划分 ----------
    DATA_START_DATE = "20170101"         # 数据起始日期
    VALID_END_DATE = "20181231"          # 验证集截止（含）— 熊市压力测试
    TRAIN_END_DATE = "20231231"          # 训练集截止（含）
    TEST_END_DATE = "20261231"           # 测试集截止（含）
    # 验证集：2017-2018（熊市） | 训练集：2019-2023 | 测试集：2024-2026

    # ---------- 训练参数 ----------
    BATCH_SIZE = 1024
    TRAIN_STEPS = 500
    MAX_FORMULA_LEN = 8

    # ---------- A股交易成本 ----------
    TOTAL_BUY_COST = 0.00025 + 0.001     # 买入成本（佣金万2.5 + 滑点千1）
    TOTAL_SELL_COST = 0.00025 + 0.0005 + 0.001  # 卖出成本（佣金+印花税千0.5+滑点千1）
    STAMP_TAX_RATE_OLD = 0.001           # 印花税率（2023-08-28 前，卖出单边千1）
    STAMP_TAX_RATE_NEW = 0.0005          # 印花税率（2023-08-28 起，卖出单边千0.5）
    STAMP_TAX_CHANGE_DATE = "20230828"   # 印花税减半生效日

    # ---------- A股交易规则 ----------
    PRICE_LIMIT_MAIN = 0.10        # 主板涨跌停 ±10%
    PRICE_LIMIT_GEM = 0.20         # 创业板/科创板 ±20%
    PRICE_LIMIT_ST = 0.05          # ST 股涨跌停 ±5%
    MIN_LOT_SIZE = 100             # 最小交易单位（股）
    MIN_TURNOVER_RATE = 0.005      # 最低换手率（过滤停牌/流动性不足）

    # ---------- 早停 ----------
    PATIENCE_LIMIT = 50           # 连续 N 步无新 best 则早停
    MIN_TRAIN_STEPS = 300         # 至少跑 N 步再允许早停（等 entropy 退火到 ≈0.06）

    # ---------- PPO ----------
    PPO_EPOCHS = 4           # 每次 rollout 的 PPO 更新轮数
    PPO_CLIP_EPS = 0.2       # PPO clip epsilon
    GRAD_CLIP_NORM = 0.5     # 梯度裁剪范数

    # ---------- 探索与多样性 ----------
    ENTROPY_COEF_START = 0.08      # 起始 entropy 系数
    ENTROPY_COEF_END = 0.01        # 终止 entropy 系数

    # ---------- 因子维度 ----------
    INPUT_DIM = 20

    # ---------- 信号输出 ----------
    SIGNAL_DIR = os.path.join(_PROJECT_ROOT, "signals")
    SIGNAL_THRESHOLD = 0.7         # sigmoid 阈值
    TOP_N_STOCKS = 20              # 截面选股数量
    REBALANCE_FREQ = 10            # 再平衡周期（交易日）：每 N 天执行一次截面选股，非再平衡日沿用上一日持仓
    REBALANCE_RANK_GAP = 5         # 换仓排名阈值：新候选领先最弱持仓至少N名才调仓（0=关闭）

    # ---------- 全市场股票池 ----------
    UNIVERSE = "all"                # "hs300" 仅沪深300 | "all" 全A股主板
    EXCLUDE_GEM = True              # 排除创业板 (300xxx, 301xxx)
    EXCLUDE_STAR = True             # 排除科创板 (688xxx)
    EXCLUDE_BSE = True              # 排除北交所 (8xxxxx)
    EXCLUDE_ST = True               # 排除 ST/*ST 股票
    IPO_MIN_DAYS = 250              # 上市最少N个自然日（过滤次新股，≈1年）

    # ---------- SFT 热启动 ----------
    SEED_FORMULA_NAMES = [         # 种子公式（名称格式，token ID 动态计算）
        ("CS_RANK", ["P_VALUE"]),                   # 价值因子截面排名
        ("CS_RANK", ["RET"]),                       # 动量截面排名
        ("CROSS", ["P_VALUE", "RET"]),              # 价值×动量交互
    ]
    SFT_STEPS = 0                  # SFT 预训练步数（0 = 关闭热启动）
