# 桩基础优化项目代码说明

## 1. 项目总体用途

这个目录包含一套风机桩基础设计、优化、验算、钢筋排布和 Abaqus 建模脚本。

主要计算链路是：

```text
基础参数配置
    -> 单目标或多目标遗传算法优化
    -> 承台与桩基受力、变形、承载力和构造验算
    -> 输出最优设计参数 JSON
    -> 生成底板、顶板和桩位布置图
    -> 使用 JSON 建立 Abaqus 桩土模型
    -> 提取 Abaqus 结果的荷载-沉降曲线
```

项目同时保留了多代优化算法、对比实验、敏感性分析和一次性修补脚本。正式使用时，应优先选择名称中没有 `legacy_` 且用途匹配的脚本。

## 2. 命名规则

重命名后的文件采用以下规则：

- `optimize_`：搜索最优设计。
- `analyze_`：计算、评估或产生分析结果。
- `verify_`：对给定方案进行完整复核。
- `render_`：生成二维钢筋或桩位布置图。
- `build_`：建立 Abaqus 模型。
- `extract_`：从 Abaqus 结果文件提取数据。
- `compare_`：运行两种算法或配置的对比实验。
- `benchmark_`：对多种算法进行统一基准测试。
- `legacy_`：历史版本或已经执行过的一次性脚本，通常不应再次运行。

## 3. 文件名称对照

| 原文件名 | 新文件名 | 文件定位 |
|---|---|---|
| `1.py` | `abaqus_postprocess/abaqus_extract_pile_load_settlement.py` | Abaqus 后处理 |
| `2.4.py` | `optimize_cost_crack_pareto.py` | 二维 Pareto 优化 |
| `4.py` | `plan_rebar_energy_path.py` | 钢筋避障路径规划 |
| `Anchor Cage Algorithm Benchmarking.py` | `benchmark_anchor_cage_algorithms.py` | 锚栓笼算法基准测试 |
| `Anchor_Optimizer.py` | `optimize_anchor_cage.py` | 锚栓笼优化 |
| `bottom_layout.py` | `render_bottom_rebar_layout.py` | 底板配筋可视化 |
| `build_pile_model.py` | `build_abaqus_pile_soil_model.py` | Abaqus 参数化建模 |
| `cap_struct_module.py` | `analyze_cap_structure.py` | 承台结构验算 |
| `config.py` | `foundation_config.py` | 全局配置 |
| `foundation_model.py` | `analyze_integrated_pile_foundation.py` | 桩基础整体计算 |
| `multi_run_and_verify.py` | `run_multistart_optimization_and_verify.py` | 多起点优化与复核 |
| `new.py` | `optimize_foundation_deb_rules.py` | Deb 可行性规则优化 |
| `Optimization.py` | `optimize_foundation_ga.py` | 当前单目标主优化器 |
| `Optimization2.1.py` | `optimize_concrete_steel_pareto.py` | 混凝土-钢材 Pareto |
| `Optimization3.py` | `optimize_three_objective_pareto.py` | 三维多目标 Pareto |
| `Pareto_Front_Analysis.py` | `analyze_anchor_pareto_front.py` | 锚栓笼 Pareto 分析 |
| `pile_struct_module.py` | `analyze_pile_structure.py` | 桩结构验算 |
| `run_compare.py` | `compare_feasible_initialization.py` | 可行初始化对比 |
| `run_compare_v2.py` | `compare_adaptive_ga.py` | 自适应遗传参数对比 |
| `run_compare_v3.py` | `compare_annealing_local_search.py` | 模拟退火局部搜索对比 |
| `Sensitivity_Analysis.py` | `analyze_anchor_sensitivity.py` | 锚栓笼参数敏感性分析 |
| `top_layout.py` | `render_top_rebar_layout.py` | 顶板配筋可视化 |
| `v2.py` | `legacy_optimize_foundation_ga.py` | 旧版单目标优化器 |
| `verification.py` | `verify_design_constraints.py` | 完整设计复核 |
| `verify_optimal.py` | `verify_hardcoded_optimal_design.py` | 固定算例复核 |
| `verify_saved_best.py` | `verify_saved_best_trial.py` | 已保存最优解复核 |
| `_fix_multi.py` | `legacy_applied_patch_multirun_serialization.py` | 已执行的临时修补 |
| `_fix_verification.py` | `legacy_applied_patch_verification_soil.py` | 已执行的临时修补 |

## 4. 每个代码文件的详细说明

### 4.1 配置与核心计算

#### `foundation_config.py`

原文件：`config.py`

作用：

- 项目的全局参数中心。
- 定义 `SoilLayer` 土层对象。
- 保存土层厚度、密度、弹性模量、黏聚力、内摩擦角、侧阻力和端阻力。
- 根据土层位置计算合理桩长范围。
- 保存设计变量范围、离散钢筋规格和 GA 参数。
- 保存极值、正常、地震和罕遇荷载工况。
- 保存混凝土材料库、钢材单价和锚栓笼参数。

输入：

- 工程地质参数。
- 设计边界条件。
- 优化变量范围和算法参数。

输出：

- 供其他模块导入的 `CONFIG`、`SOIL_LAYERS` 等全局对象。

注意事项：

- 修改它会影响整个项目的荷载、土层、变量范围和材料取值。
- 该文件不应包含具体优化流程，只负责提供配置数据。

#### `analyze_integrated_pile_foundation.py`

原文件：`foundation_model.py`

作用：

- 桩基础整体物理模型。
- 将上部竖向力、弯矩、水平力和扭矩分配到各根桩。
- 考虑承台自重、覆土、地下水和桩位坐标。
- 计算承台混凝土体积、钢材重量和等效体积造价。
- 返回每根桩的轴力、水平力、设计反力和详细受力结果。
- 提供 `assess_design_performance()` 作为优化器的总体性能入口。

核心类：

- `IntegratedPileFoundation`

核心方法：

- `get_quantities()`：计算工程量和造价相关数据。
- `get_pile_reactions_detailed()`：计算各桩详细反力。
- `get_individual_pile_forces()`：返回单桩受力列表。

输入：

- 完整设计参数。
- 荷载工况。
- 土层列表。

输出：

- 承载力、沉降、受力、工程量和违反约束程度。

#### `analyze_cap_structure.py`

原文件：`cap_struct_module.py`

作用：

- 承台结构验算模块。
- 采用单位条带法计算承台径向、环向受力。
- 验算正截面抗弯、斜截面抗剪、裂缝宽度和冲切承载力。
- 同时考虑边缘底板、斜面段和台柱区域的厚度变化。

核心类：

- `CapStructuralAnalysis`

主要验算：

- 弯曲配筋验算。
- 斜截面抗剪验算。
- 裂缝宽度验算。
- 角桩和台柱冲切验算。

#### `analyze_pile_structure.py`

原文件：`pile_struct_module.py`

作用：

- 单桩和群桩结构分析核心模块。
- 计算桩身抗压、抗拔和水平承载力。
- 验算桩身强度、局部承压、压屈稳定性和抗拉强度。
- 计算桩身弯矩、剪力、裂缝宽度。
- 检查软弱下卧层和负摩阻力。
- 计算基础沉降和整体倾斜。
- 生成并查询圆形截面的 P-M 相互作用曲线。

核心类：

- `PileStructuralAnalysis`
- `PMInteractionManager`

主要用途：

- 为优化器提供桩基约束。
- 为最终 17 项复核报告提供计算能力。

### 4.2 单目标优化器

#### `optimize_foundation_ga.py`

原文件：`Optimization.py`

定位：

- 当前最重要的单目标主优化器。

作用：

- 使用 DEAP 遗传算法搜索承台、桩位和配筋方案。
- 使用拉丁超立方或可行初始化生成初始种群。
- 使用多点交叉、多项式变异和锦标赛选择。
- 使用 16 个进程并行评估个体。
- 在 GA 中加入模拟退火式局部搜索。
- 根据种群收敛情况动态调整交叉率和变异率。
- 将成本和约束违反度组合成最终适应度。
- 输出收敛曲线、指标曲线和桩位布置图。
- 输出 `abaqus_postprocess/abaqus_pile_model.json`。
- 输出 `Optimized_Rebar_Data.json`。
- 生成底板和顶板钢筋避让图。

关键函数：

- `decode_gene()`：将基因转换成几何、桩和配筋参数。
- `feasible_init_population()`：用快速规则筛选可行初始方案。
- `calculate_crowding_penalty()`：检查钢筋与锚栓套管的拥挤和碰撞。
- `evaluate()`：调用物理模型并返回成本和违反度。
- `sa_local_search()`：对候选方案进行局部改进。
- `compute_adaptive_probs()`：动态计算交叉和变异概率。
- `export_abaqus_json()`：导出 Abaqus 建模数据。

输入：

- `foundation_config.py` 中的配置。
- 用户输入的“是否允许受拉桩”和“是否固定桩长”。

输出：

- 最优设计参数和终端报告。
- `best_fitness_curve.png`
- `optimization_metrics.png`
- `pile_layout.png`
- `abaqus_postprocess/abaqus_pile_model.json`
- `Optimized_Rebar_Data.json`

#### `legacy_optimize_foundation_ga.py`

原文件：`v2.py`

作用：

- 当前主优化器的旧版本。
- 已包含多进程、等效体积造价、钢筋避让和 Abaqus JSON 导出。
- 缺少当前版本中的可行初始化、自适应参数和模拟退火局部搜索。

使用建议：

- 只用于复现旧结果或做版本对比。
- 新计算应使用 `optimize_foundation_ga.py`。

#### `optimize_foundation_deb_rules.py`

原文件：`new.py`

作用：

- 使用 Deb 可行性规则处理约束的单目标遗传算法实验版。
- 优先选择可行解；同为可行解时比较成本；同为不可行解时比较违反程度。
- 包含业务计算、配筋检查、优化主循环和结果保存。

使用建议：

- 这是单目标优化路线中的实验版本。
- 如果需要研究约束选择策略，可以用它与当前主优化器对比。

### 4.3 多目标优化器

#### `optimize_concrete_steel_pareto.py`

原文件：`Optimization2.1.py`

作用：

- 使用 NSGA-II 搜索混凝土用量和钢材用量之间的 Pareto 前沿。
- 将离散钢筋根数改为连续搜索区间。
- 减少取整带来的搜索断层。
- 对 Pareto 前沿执行非支配排序。
- 自动寻找权衡前沿上的拐点。
- 输出混凝土用量、钢材用量和受力指标。

输出：

- `pareto_front_with_stars.png`

注意事项：

- 该版本不导出 Abaqus JSON。

#### `optimize_cost_crack_pareto.py`

原文件：`2.4.py`

作用：

- 使用 NSGA-II 搜索“材料成本”和“最大裂缝宽度”之间的 Pareto 前沿。
- 将目标从单纯材料用量升级为经济性和耐久性权衡。
- 执行归一化，避免百万级成本与毫米级裂缝在排序中失衡。
- 自动寻找膝点方案。
- 输出极致耐久、膝点、极致省钱等代表方案。

输出：

- `pareto_front_crack_cost.png`
- 带时间戳的 Abaqus 参数 JSON。

#### `optimize_three_objective_pareto.py`

原文件：`Optimization3.py`

作用：

- 三维多目标优化器。
- 同时最小化混凝土用量、钢材用量和基础占地面积。
- 使用 DEAP 三维非支配排序。
- 提取三维 Pareto 前沿。
- 输出代表方案和 Abaqus 建模 JSON。

输出：

- `pareto_front_3d.png`
- 带时间戳的 Abaqus 参数 JSON。

### 4.4 配筋和桩位可视化

#### `render_bottom_rebar_layout.py`

原文件：`bottom_layout.py`

作用：

- 底板径向钢筋和环向钢筋排布可视化。
- 处理通长筋、加强筋和外插筋。
- 根据锚栓套管位置推动钢筋角度，避免碰撞。
- 使用 Matplotlib 输出最终布置图。

核心类：

- `BottomRebarLayout`

#### `render_top_rebar_layout.py`

原文件：`top_layout.py`

作用：

- 顶板径向钢筋和环向钢筋排布可视化。
- 处理台柱、中心留空区和锚栓套管。
- 生成钢筋避让后的角度。

核心类：

- `TopRebarLayout`

#### `plan_rebar_energy_path.py`

原文件：`4.py`

作用：

- 独立的钢筋平滑避障算法。
- 将钢筋离散成多个路径点。
- 使用弯曲能和障碍物排斥能构造目标函数。
- 使用 SciPy 的 `L-BFGS-B` 优化路径。
- 生成钢筋绕开障碍物的平滑曲线。

定位：

- 偏研究性工具，不属于主优化流程的必需模块。

### 4.5 Abaqus 建模和后处理

#### `build_abaqus_pile_soil_model.py`

原文件：`build_pile_model.py`

作用：

- 在 Abaqus 中参数化建立单桩-多层土相互作用模型。
- 读取 `abaqus_postprocess/abaqus_pile_model.json`。
- 创建混凝土、钢筋和土层材料。
- 引入混凝土损伤塑性 CDP 模型。
- 建立桩土接触、地应力分析步和 ULS 加载步。
- 划分网格并创建 Abaqus Job。

输入：

- `abaqus_postprocess/abaqus_pile_model.json`

运行环境：

- 必须在 Abaqus Python 环境或 Abaqus CAE 中运行。
- 普通 Python 环境没有 `abaqus` 和 `odbAccess` 模块。

#### `abaqus_postprocess/abaqus_extract_pile_load_settlement.py`

原文件：`1.py`

作用：

- 打开 Abaqus ODB 结果文件。
- 查找坐标位于 `(0, 0, 0)` 的桩顶参考点。
- 提取每个分析帧的竖向位移 `U3`。
- 将位移转换成毫米。
- 按照最大荷载乘分析步时间计算桩顶荷载。
- 输出荷载-沉降数据表。

输入：

- `Run_Pile_Soil_Interaction_ULS.odb`

输出：

- 终端中的 `Step_Time`、`Settlement(mm)` 和 `Top_Load(kN)` 表格。

注意事项：

- 当前荷载按照 `MAX_LOAD_KN * time` 线性计算。
- 如果模型使用非线性幅值，应从反力 `RF3` 提取实际荷载。
- 当前使用 `abs(U3)`，不能区分沉降和上拔。

### 4.6 完整验算与复核

#### `verify_design_constraints.py`

原文件：`verification.py`

作用：

- 对给定设计执行独立的 17 项安全复核。
- 计算承台和桩基的规范利用率。
- 处理偏航扭矩分配到单桩产生的水平剪力。
- 按桩圈提取最大压力和最大拉力。
- 复核抗压、抗拔、水平承载、结构强度、裂缝、沉降、倾斜等指标。

核心函数：

- `generate_ring_coords()`：生成圆周桩位坐标。
- `run_full_verification()`：执行全部复核并输出结果。

定位：

- 该模块用于最终验收，不应被 GA 当作快速评价函数。

#### `run_multistart_optimization_and_verify.py`

原文件：`multi_run_and_verify.py`

作用：

- 使用多个随机种子重复运行 GA。
- 每次都保存完整参数、成本和违反度。
- 从所有试验中筛选最佳可行解。
- 将最佳解转换成完整验算数据。
- 调用 `verify_design_constraints.py` 做最终复核。

输出：

- `multi_run_results/trial_XX.json`
- `multi_run_results/summary.json`

定位：

- 提高遗传算法可靠性的多起点实验入口。

#### `verify_hardcoded_optimal_design.py`

原文件：`verify_optimal.py`

作用：

- 使用一组写死在脚本中的最优参数。
- 计算承台工程量、桩数和桩身参数。
- 调用完整验算函数。
- 适合复现某一次特定优化结果。

注意事项：

- 参数写死在源码中，换工程时必须逐项修改。

#### `verify_saved_best_trial.py`

原文件：`verify_saved_best.py`

作用：

- 扫描 `multi_run_results/trial_*.json`。
- 自动选择成本最低的可行解。
- 将保存的简化参数转换为完整验算格式。
- 重新运行 17 项复核，不需要重跑 GA。

定位：

- 多起点优化完成后最方便的结果复核入口。

### 4.7 锚栓笼优化与分析

#### `optimize_anchor_cage.py`

原文件：`Anchor_Optimizer.py`

作用：

- 优化双圈锚栓笼的 PCD、锚栓数量和锚栓规格。
- 读取 `Optimized_Rebar_Data.json` 中的基础边界条件。
- 计算 ULS、疲劳应力、台柱空间和钢筋净距。
- 对不可行方案施加惩罚。
- 输出最优锚栓笼布置。
- 调用底板和顶板配筋可视化模块。

输入：

- `Optimized_Rebar_Data.json`

#### `benchmark_anchor_cage_algorithms.py`

原文件：`Anchor Cage Algorithm Benchmarking.py`

作用：

- 建立锚栓笼优化基准问题。
- 对比网格搜索、粒子群算法 PSO 和遗传算法 GA。
- 统计最佳目标值、计算次数和运行时间。
- 生成算法性能对比图。

定位：

- 算法研究或选型工具，不参与正式基础设计。

#### `analyze_anchor_pareto_front.py`

原文件：`Pareto_Front_Analysis.py`

作用：

- 对双圈锚栓笼执行 NSGA-II 多目标优化。
- 同时考虑总造价和疲劳应力裕量。
- 输出 Pareto 前沿图。
- 分析成本与耐久性之间的权衡。

#### `analyze_anchor_sensitivity.py`

原文件：`Sensitivity_Analysis.py`

作用：

- 对锚栓规格、锚栓数量、PCD、台柱半径和承台厚度进行参数敏感性分析。
- 多次执行锚栓笼优化。
- 比较不同输入条件下的最优成本和最优方案。
- 输出适合论文或设计报告使用的折线图。

### 4.8 对比实验脚本

#### `compare_feasible_initialization.py`

原文件：`run_compare.py`

作用：

- 对比原始 LHS 初始化和“可行初始化”。
- 使用相同 GA 参数分别运行。
- 输出成本收敛曲线和最终结果。

输出：

- `improvement_1_compare.png`

#### `compare_adaptive_ga.py`

原文件：`run_compare_v2.py`

作用：

- 对比固定交叉/变异概率和自适应交叉/变异概率。
- 两种方法都使用可行初始化。
- 输出收敛速度和最终质量对比。

输出：

- `improvement_2_compare.png`

#### `compare_annealing_local_search.py`

原文件：`run_compare_v3.py`

作用：

- 对比纯 GA 与“GA + 模拟退火局部搜索”。
- 检查局部搜索能否进一步降低成本和违反度。

输出：

- `improvement_3_compare.png`

### 4.9 历史和一次性维护脚本

#### `legacy_optimize_foundation_ga.py`

原文件：`v2.py`

作用：

- 旧版单目标优化器。
- 适合代码考古或复现历史结果。

#### `legacy_applied_patch_multirun_serialization.py`

原文件：`_fix_multi.py`

作用：

- 一次性修改多起点结果序列化逻辑。
- 使 `ring_opt` 和 `top_ring_opt` 能写入 JSON。

使用建议：

- 该修补大概率已经执行。
- 当前源码修改后不应再次运行，否则断言可能失败。

#### `legacy_applied_patch_verification_soil.py`

原文件：`_fix_verification.py`

作用：

- 一次性修改验算模块的有效土层处理。
- 扣除承台埋深以上的土，使复核口径与优化器一致。

使用建议：

- 该修补大概率已经执行。
- 当前源码修改后不应再次运行。

## 5. 推荐运行顺序

### 5.1 单目标设计

```text
optimize_foundation_ga.py
    -> abaqus_postprocess/abaqus_pile_model.json
    -> Optimized_Rebar_Data.json
    -> build_abaqus_pile_soil_model.py
    -> Abaqus Job
    -> abaqus_postprocess/abaqus_extract_pile_load_settlement.py
```

### 5.2 多起点可靠性验证

```text
run_multistart_optimization_and_verify.py
    -> multi_run_results/trial_XX.json
    -> verify_saved_best_trial.py
```

### 5.3 完整设计复核

```text
verify_hardcoded_optimal_design.py
```

或者：

```text
verify_saved_best_trial.py
```

### 5.4 多目标方案筛选

```text
optimize_concrete_steel_pareto.py
```

或者：

```text
optimize_cost_crack_pareto.py
```

或者：

```text
optimize_three_objective_pareto.py
```

### 5.5 锚栓笼设计

```text
optimize_anchor_cage.py
```

需要算法研究时再运行：

```text
benchmark_anchor_cage_algorithms.py
analyze_anchor_pareto_front.py
analyze_anchor_sensitivity.py
```

## 6. 维护建议

1. `optimize_foundation_ga.py` 是当前正式单目标入口，后续修复应优先针对它。
2. `v2.py` 和 `new.py` 仅作为历史算法版本，不建议继续叠加新功能。
3. 三个 `run_compare` 实验依赖旧路径 `C:\Users\18445\...`，迁移电脑后需要修改路径。
4. `build_abaqus_pile_soil_model.py` 和 `abaqus_postprocess/abaqus_extract_pile_load_settlement.py` 必须在 Abaqus Python 环境运行。
5. 不应再次运行两个 `legacy_applied_patch_*` 文件。
6. 修改 `foundation_config.py` 后，应重新运行完整验算，不能只检查优化器能够启动。
