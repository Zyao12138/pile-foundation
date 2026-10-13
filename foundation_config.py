# -*- coding: utf-8 -*-
"""
模块名称: foundation_config.py
功能: 基础参数配置中心 (已整合原 geo_tools 地质计算功能)
"""
import multiprocessing

# ==============================================================================
# 1. 基础类定义与地质工具函数
# ==============================================================================
class SoilLayer:
    def __init__(self, name, thickness, density, Es, f_ak, n_b, n_d, v, Ed, C, angle,
                 qsik=0, qpk=0, m_value=10, xi_n=0.3):
        self.name = name
        self.thickness = thickness
        self.gamma = density
        self.Es = Es
        self.v = v
        self.Ed = Ed
        self.f_ak = f_ak
        self.n_b = n_b
        self.n_d = n_d
        self.C = C
        self.angle = angle
        self.qsik = qsik
        self.qpk = qpk
        self.m_value = m_value
        self.xi_n = xi_n
        self.is_liquefiable = False
        self.spt_ratio = 1.0


def calculate_length_range(soil_layers, target_layer_idx, min_embedment, max_embedment=None, safety_margin=0.5):
    """
    根据土层信息动态计算桩长优化范围。
    兼容 soil_layers 为字典列表或对象列表。
    """
    # 1. 基础校验
    if not soil_layers:
        raise ValueError("土层数据为空")
    if target_layer_idx < 0 or target_layer_idx >= len(soil_layers):
        raise ValueError(f"持力层索引 {target_layer_idx} 超出范围 (共 {len(soil_layers)} 层)")

    # 2. 遍历累加深度
    current_depth = 0.0
    bearing_top_depth = 0.0
    bearing_bottom_depth = 0.0

    for i, layer in enumerate(soil_layers):
        # 兼容字典和对象属性访问
        if isinstance(layer, dict):
            thickness = layer['thickness']
        else:
            thickness = layer.thickness

        if i == target_layer_idx:
            bearing_top_depth = current_depth
            bearing_bottom_depth = current_depth + thickness
            break
        current_depth += thickness

    # 3. 计算下限
    calc_min = bearing_top_depth + min_embedment

    # 4. 计算上限
    limit_physical = bearing_bottom_depth - safety_margin

    if max_embedment is not None:
        limit_design = bearing_top_depth + max_embedment
        calc_max = min(limit_physical, limit_design)
    else:
        calc_max = limit_physical

    # 5. 异常处理
    if calc_min > calc_max:
        print(f"[GeoTools 警告] 持力层过薄，自动修正上限。")
        calc_max = calc_min + 0.5

    return round(calc_min, 2), round(calc_max, 2)


# ==============================================================================
# 2. 土层数据 (根据 ZK5 钻孔柱状图修改)
# ==============================================================================
# 参数说明: Name, Thickness(m), Density(kN/m3), Es(MPa), Fak(kPa), nb, nd, nu, Ed(MPa), C(kPa), Phi(deg), qsik(kPa), qpk(kPa), m(MN/m4), xi_n
SOIL_LAYERS = [
    # --------------------------------------------------------------------------
    # 第1层: ① 耕土 (根据图片：0.0-0.5m)
    # --------------------------------------------------------------------------
    SoilLayer(
        name="① 耕土",
        thickness=0.5,      # [修改] 厚度 0.5m
        density=17.5,       # 杂填土/耕土重度稍低
        Es=2.5,             # 压缩模量低
        f_ak=60,            # 承载力低
        n_b=0, n_d=1.0,
        v=0.35,
        Ed=8.0,
        C=10.0, angle=10.0,
        qsik=0,             # 表层土通常不计侧阻
        qpk=0,
        m_value=2,          # 软弱土
        xi_n=0.0
    ),

    # --------------------------------------------------------------------------
    # 第2层: ② 粉质黏土 (根据图片：0.5-6.5m，硬塑，SPT≈16-19)
    # --------------------------------------------------------------------------
    SoilLayer(
        name="② 粉质黏土",
        thickness=6.0,      # [修改] 厚度 6.0m
        density=19.2,       # 硬塑状态，重度尚可
        Es=7.5,             # 估算 Es (N*0.4~0.5)
        f_ak=180,           # 结合 SPT=17 估算
        n_b=0.3, n_d=1.6,
        v=0.32,
        Ed=18.0,
        C=25.0, angle=18.0,
        qsik=50,            # [估算] 硬塑粉质黏土 (JGJ94)
        qpk=0,
        m_value=12,         # 中等硬度
        xi_n=0.30
    ),

    # --------------------------------------------------------------------------
    # 第3层: ③ 黏土 (根据图片：6.5-23.1m，硬塑，SPT≈21-30)
    # --------------------------------------------------------------------------
    SoilLayer(
        name="③ 黏土",
        thickness=16.6,     # [修改] 厚度 16.6m
        density=19.8,       # 深度增加，压实度增加
        Es=12.0,            # SPT均值约25，模量较高
        f_ak=240,           # 承载力较好
        n_b=0.3, n_d=1.6,
        v=0.30,
        Ed=25.0,
        C=45.0, angle=20.0,
        qsik=70,            # [估算] SPT=25左右的黏土
        qpk=0,
        m_value=20,         # 硬土
        xi_n=0.30
    ),

    # --------------------------------------------------------------------------
    # 第4层: ④ 黏土 (根据图片：23.1-35.0m，硬塑/坚硬，SPT≈33-34)
    # --------------------------------------------------------------------------
    SoilLayer(
        name="④ 黏土",
        thickness=11.9,     # [修改] 厚度 11.9m
        density=20.5,       # 深层高密实度
        Es=16.0,            # SPT>30，高压缩模量
        f_ak=320,           # 高承载力
        n_b=0.3, n_d=1.6,
        v=0.28,
        Ed=35.0,
        C=55.0, angle=22.0,
        qsik=85,            # [估算] 坚硬黏土侧阻力高
        qpk=2200,           # [估算] 可作为桩端持力层，端阻力估值
        m_value=30,         # 坚硬土
        xi_n=0.35
    )
]

# ==============================================================================
# 3. 动态计算桩长 (支持自动识别持力层 + 手动指定)
# ==============================================================================
# 判断当前是否为主进程 (防止多进程下子进程疯狂重复打印)
IS_MAIN_PROCESS = multiprocessing.current_process().name == 'MainProcess'

# --- [用户设置] 持力层指定 ---
MANUAL_TARGET_LAYER = None

# --- [逻辑处理] ---
if MANUAL_TARGET_LAYER is not None:
    if 0 <= MANUAL_TARGET_LAYER < len(SOIL_LAYERS):
        TARGET_LAYER_INDEX = int(MANUAL_TARGET_LAYER)
        if IS_MAIN_PROCESS:
            print(f"--- [Config] 已手动指定持力层: {SOIL_LAYERS[TARGET_LAYER_INDEX].name} (Index={TARGET_LAYER_INDEX}) ---")
    else:
        raise ValueError(f"手动指定的层索引 {MANUAL_TARGET_LAYER} 超出土层范围")
else:
    max_qpk = -1.0
    best_idx = -1
    for i, layer in enumerate(SOIL_LAYERS):
        if layer.qpk >= max_qpk:
            max_qpk = layer.qpk
            best_idx = i

    if best_idx != -1 and max_qpk > 0:
        TARGET_LAYER_INDEX = best_idx
        if IS_MAIN_PROCESS:
            print(f"--- [Config] 自动识别最佳持力层: {SOIL_LAYERS[best_idx].name} (Index={best_idx}, qpk={max_qpk}kPa) ---")
    else:
        TARGET_LAYER_INDEX = len(SOIL_LAYERS) - 1
        if IS_MAIN_PROCESS:
            print(f"--- [Config] 未检测到明显端阻层，默认选最底层作为持力层: {SOIL_LAYERS[TARGET_LAYER_INDEX].name} ---")

# --- [计算桩长范围] ---
MIN_EMBED = 2.0

try:
    # 这里直接调用刚刚合并进来的函数，不再需要 geo_tools. 前缀
    min_L, max_L = calculate_length_range(
        soil_layers=SOIL_LAYERS,
        target_layer_idx=TARGET_LAYER_INDEX,
        min_embedment=MIN_EMBED
    )
    if IS_MAIN_PROCESS:
        print(f"--- [Config] 桩长优化范围计算成功: {min_L}m ~ {max_L}m ---")
except Exception as e:
    if IS_MAIN_PROCESS:
        print(f"--- [Config] 桩长计算出错: {e}, 使用默认备用值 ---")
    depth_top = sum([l.thickness for i, l in enumerate(SOIL_LAYERS) if i < TARGET_LAYER_INDEX])
    min_L = int(depth_top + MIN_EMBED)
    max_L = int(depth_top + SOIL_LAYERS[TARGET_LAYER_INDEX].thickness - 1.0)
    if IS_MAIN_PROCESS:
        print(f"--- [Config] 使用备用范围: {min_L}m ~ {max_L}m ---")


# ==============================================================================
# 4. 优化变量配置 & 常量库
# ==============================================================================
CONFIG = {
    "REBAR_RELATIONS": {
        "底板短筋数量 (n2)": "等于 底板通长筋数量 (n1)",
        "底板外插筋数量 (n3)": "等于 底板通长筋数量 (n1)",
        "底板外插筋直径 (d3)": "等于 底板通长筋直径 (d1)",
        "顶板外插筋数量 (n5)": "等于 顶板通长筋数量 (n4)",
        "顶板外插筋直径 (d5)": "等于 顶板通长筋直径 (d4)"
    },

    # --- [核心修改] 构造设计参数 ---
    "DESIGN_PARAMS": {
        'r1': 1.5,   # 中心不配筋区域半径 (m)
        's1': 150,   # 底板环向筋间距 (mm)
        's1_2': 200, # 底板附加层环向筋间距 (mm)
        's2': 120,   # 顶板环向筋间距 (mm)

        # 竖向拉筋/架立筋配置
        "tie_bar": {
            "d": 0.016,  # 直径 16mm
            "sx": 0.6,   # X方向间距 0.6m
            "sy": 0.6    # Y方向间距 0.6m
        },

        # --- [新增] 单桩承载力硬性控制指标 ---
        "LIMITS": {
            'MIN_Ra': 2200.0,  # 抗压最小值 (kN)
            'MIN_Rt': 1000.0,  # 抗拔最小值 (kN)
            'MIN_Rh': 70.0     # 水平最小值 (kN)
        },
    },

    "ANCHOR_CAGE": {
        "PCD": 4.8,             # 锚栓分布圆直径 (m)
        "D_outer": 5.1,         # 锚栓笼整体最大外径 (m)
        "L_total": 3.4,         # 锚栓总长度 (m)
        "top_expose": 0.3,      # 锚栓顶部外露长度(穿法兰+张拉空间) (m)
        "bottom_groove": 0.3,   # 承台底部的锚栓预留槽深度 (m)
        "bottom_cover": 0.2,    # 锚栓笼底部的混凝土保护层厚度 (m)
        "weight_ton": 35.0,      # 锚栓笼及支撑件总自重估算 (吨)
        "N_BOLTS": 180,         # 锚栓笼的总锚栓数量 (根)
        "BOLT_D": 48            # 单根锚栓的直径 (mm，即 M48)
    },

    # --- 22变量外层 + 4变量锚栓笼内层优化 ---
    # MODE="fixed" 用于可比较基线；MODE="nested" 启用内层确定性搜索。
    "ANCHOR_OPT": {
        "MODE": "fixed",
        "RESEARCH_PARAMETERS_PENDING_CALIBRATION": [
            "公开资料研究基线，待厂家塔底接口校准", "疲劳弯矩比例为敏感性参数",
            "35d锚固长度", "50MPa疲劳限值", "构造净距与保护层",
            "10.9级材料模型", "预紧力与法兰刚度尚未显式建模",
            "锚栓与法兰单价及简化受力公式"
        ],
        # 仅作算法研究的默认场景。正式设计必须使用风机厂家疲劳载荷谱。
        "FATIGUE_MOMENT_RATIO": 0.10,
        "FATIGUE_MOMENT_RATIO_SCENARIOS": [0.05, 0.10, 0.15, 0.20, 0.30],
        "BASELINE": {
            "PCD_in_m": 5.8,
            "PCD_out_m": 6.3,
            "N_per_ring": 90,
            "bolt_type": "M56",
        },
        "VARIABLES": {
            "PCD_IN_MIN_M": 5.2,
            "PCD_OUT_MAX_M": 7.0,
            "PCD_STEP_M": 0.05,
            "N_PER_RING_VALUES": list(range(72, 109, 4)),
            "BOLT_TYPES": ["M48", "M56", "M64", "M72"],
            # 内外圈长度保持一致；4.0 m为原模型基准值。
            "L_TOTAL_VALUES_M": [3.4, 3.6, 3.8, 4.0, 4.2, 4.4],
        },
        "BOLT_CATALOG": {
            "M42": {"d_mm": 42.0, "pitch_mm": 4.5, "sleeve_d_mm": 55.0, "unit_cost_rmb": 1200.0},
            "M48": {"d_mm": 48.0, "pitch_mm": 5.0, "sleeve_d_mm": 65.0, "unit_cost_rmb": 1800.0},
            "M56": {"d_mm": 56.0, "pitch_mm": 5.5, "sleeve_d_mm": 75.0, "unit_cost_rmb": 2600.0},
            "M64": {"d_mm": 64.0, "pitch_mm": 6.0, "sleeve_d_mm": 85.0, "unit_cost_rmb": 3500.0},
            # M72价格及套管尺寸为研究估值，需由供应商报价和产品图校准。
            "M72": {"d_mm": 72.0, "pitch_mm": 6.0, "sleeve_d_mm": 95.0, "unit_cost_rmb": 4500.0},
        },
        "MATERIAL": {
            "F_T_DESIGN_MPA": 500.0,
        },
        "FIXED": {
            "ANCHORAGE_FACTOR_D": 35.0,
            "BOTTOM_COVER_M": 0.2,
            "L_TOTAL_M": 4.0,
            "TOP_EXPOSE_M": 0.3,
            "BOTTOM_GROOVE_M": 0.3,
        },
        "CONSTRAINTS": {
            "R_VOID_M": 1.5,
            "MIN_EDGE_COVER_M": 0.25,
            "EXTRA_RADIAL_CLEARANCE_M": 0.10,
            "EXTRA_CIRC_CLEARANCE_MM": 50.0,
            "ALLOWABLE_FATIGUE_STRESS_MPA": 50.0,
            "ENVELOPE_ALLOWANCE_M": 0.0,
        },
        "COST": {
            "FLANGE_FACTOR": 3.0,
            "FLANGE_COST_PER_M_RMB": 15000.0,
            "UNIT_COST_REFERENCE_LENGTH_M": 4.0,
        },
        "SOLVER": {
            # 基线 + 4种规格 × 10种根数 × 6种长度 × 每组最多8个双PCD候选
            "MAX_CANDIDATES": 1921,
            "PCD_PAIRS_PER_COMBO": 8,
            "FULL_PCD_GRID": False,
            "CACHE_MAX_ENTRIES": 2048,
            "CACHE_ENABLED": True,
        },
    },

    # --- 真双层优化变量划分：12个外层体系变量 + 15个内层细部变量 ---
    "BILEVEL_OPT": {
        "VERSION": "outer12_inner15_anchor5_v2",
        "OUTER_INDICES": [0, 1, 2, 3, 4, 5, 14, 15, 16, 17, 18, 19],
        "INNER_REBAR_INDICES": [6, 7, 8, 9, 10, 11, 12, 13, 20, 21],
        "INNER_ANCHOR_VARIABLES": ["PCD_in_m", "PCD_out_m", "N_per_ring", "bolt_type", "L_total_m"],
        "INNER_METHOD": "deterministic_coordinate_then_anchor_refinement",
        "COORDINATE_PASSES": 2,
        "SEED_FRACTIONS": [0.25, 0.50, 0.75],
        "PCD_SAMPLES_DURING_GA": 8,
        "FINAL_FULL_GRID_RECHECK": True,
    },

    "var_ranges": [
        # --- 0-5: 承台几何 #
        (800, 1300),  # [0] R1-基础底板半径
        (200, 300),   # [1] H-埋深
        (90, 150),    # [2] h1-底板边缘高度
        (120, 220),   # [3] h2-棱台高度
        (50, 120),    # [4] h3-台柱高度
        (360, 500),   # [5] r-台柱半径

        # --- 6-9: 底板配筋 ---
        [100, 110, 120, 130, 140, 150, 160, 180, 200], # [6] n1-底板通长筋数量
        [22, 25, 28, 32],     # [7] d1-底板通长筋直径
        [20, 22, 25, 28, 32], # [8] d2-底板短筋直径
        [20, 22, 25, 28, 32], # [9] D1-底板环向钢筋直径
        [20, 22, 25, 28, 32], # [10] D1_2-底板附加层环筋直径


        # --- 10-12: 顶板配筋 ---
        [100, 110, 120, 130, 140, 150], # [10] n4-顶板通长筋数量
        [22, 25, 28, 32],     # [11] d4-顶板通长筋直径
        [20, 22, 25, 28, 32], # [12] D2-顶板环向钢筋直径


        # --- 13-20: 桩基部分 ---
        (min_L, max_L),       # [13] 桩长 L
        [80,90,100,120],                 # [14] 桩径 D
        [8, 10, 12, 14, 16],      # [15] m1-内圈桩数#
        [16, 18, 20, 22, 24, 26, 28,32,36], # [16] m2-外圈桩数
        (45, 70),             # [17] 内圈半径比例
        (78, 91),             # [18] 外圈半径比例
        [12, 14, 16, 18, 20, 22, 24, 26, 28, 32, 36, 40], # [19] 桩身主筋根数
        [14, 16, 18, 20, 22, 25, 28, 32]                  # [20] 桩身主筋直径
    ],

    # --- 荷载工况 ---
    "load_cases": {
        "Extreme": {"FZ": 7918.5, "MXY": 185957.1, "FXY": 1389.4, "Mz": 1644.2},
        "Normal":  {"FZ": 7914.5, "MXY": 124139.8, "FXY": 878.5,  "Mz": 2004.7},
        "Seismic": {"FZ": 7954.6, "MXY": 133180.5, "FXY": 1071.3, "Mz": 26.8},
        "Rare":    {"FZ": 7924.5, "MXY": 141063.5, "FXY": 1239.6, "Mz": 3981.2}
    },

    # --- 荷载组合 ---
    "LOAD_COMBINATIONS": [
        {"name": "ULS_Normal",  "type": "ULS",       "case_ref": "Extreme", "gamma_G": 1.30, "gamma_Q": 1.5},
        {"name": "ULS_Seismic", "type": "ULS",       "case_ref": "Seismic", "gamma_G": 1.20, "gamma_Q": 1.3},
        {"name": "SLS_Normal",  "type": "SLS",       "case_ref": "Normal",  "gamma_G": 1.00, "gamma_Q": 1.0},
        {"name": "SLS_Crack",   "type": "SLS_Crack", "case_ref": "Normal",  "gamma_G": 1.00, "gamma_Q": 1.0}
    ],

    # --- 锚固参数 ---
    "ANCHOR_PARAMS": {
        'BOLT_D': 48,
        'FYK_STEEL': 360
    },

    # --- 约束限值 ---
    "CONSTRAINTS": {
        'MIN_NET_SPACING': 50.0
    },

    # --- 综合单价 ---
    "UNIT_PRICES": {
        'STEEL': 5500.0,
        'CONCRETE': 500.0
    },

    # --- 桩基常量 ---
    "PILE_CONSTANTS": {
        'STIRRUP_D': 0.012,
        'STIRRUP_S_DENSE': 0.1,
        'STIRRUP_S_NORM': 0.2
    },
    "pile_geo": {
        "cover": 0.06
    },

    # --- 遗传算法参数 ---
    "GA_PARAMS": {
        'POP_SIZE': 1000,
        'N_GEN': 500,
        'CX_PROB': 0.7,
        'MUT_PROB': 0.3,
        'TOURN_SIZE': 2,
        'ETA': 10.0,
        'ELITE_RATIO': 0.01
    },

    # --- 混凝土材料库 ---
    "CONCRETE_LIB": [
        {'name': 'C35', 'fc': 16.7, 'ft': 1.57, 'ftk': 2.20, 'Ec': 3.15e4, 'price': 480},
        {'name': 'C40', 'fc': 19.1, 'ft': 1.71, 'ftk': 2.39, 'Ec': 3.25e4, 'price': 500},
    ]
}
