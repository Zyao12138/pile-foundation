# 风机基础22+4嵌套优化

已合并至当前项目。外层仍为原22个变量，`CONFIG["var_ranges"]`未改变；独立`CONFIG["ANCHOR_OPT"]`控制锚栓搜索。完整实测结果见[验证报告](experiments_anchor_22_plus_4/REPORT_CN.md)。

## 入口

```powershell
python experiment_runner.py --variant fixed_single --anchor-mode fixed --seeds 0 1 --pop-size 8 --generations 2 --cores 1 --output-dir experiments_anchor_22_plus_4
python experiment_runner.py --variant nested_single --anchor-mode nested --seeds 0 1 --pop-size 8 --generations 2 --cores 1 --output-dir experiments_anchor_22_plus_4
```

双进程测试将`--cores`改为2，并分别用`fixed_multi`、`nested_multi`命名。未执行1000×500正式规模。程序调用可用`run_optimization(anchor_mode="nested", ...)`；交互main入口默认仍为fixed。

## 实现

- `foundation_config.py`：独立基线、4变量网格、规格目录、成本、构造参数和预算；22项外层范围保持原样。
- `anchor_cage_solver.py`：纯函数`evaluate_anchor_design`分别返回真实人民币成本、可行性、总违反度及逐项约束。`solve_anchor_for_foundation`负责确定性搜索、剪枝和缓存。
- `optimize_foundation_ga.py`：候选通过`params["anchor"]`传入；内层调用完整联动评价，先保证可行性，再比较基础与锚栓总成本；全部不可行时按违反度、总成本依次选择。不修改全局候选锚栓配置。
- `analyze_integrated_pile_foundation.py`：钢筋有效长度和瓶颈净距跟随当前候选外包尺寸；统一`calculate_design_cost`。主评价同时联动台柱尺寸、桩位净距及两圈套管的局部拥挤校验。
- `experiment_runner.py`：`--anchor-mode fixed|nested_2|nested`分别对应固定锚栓、仅优化规格数量、完整22+4；保存`best_anchor`、成本分项、逐项约束、参数快照和待校准项；汇总包含可行运行率和最终种群可行率。
- `test_anchor_nested.py`：6项回归验证，包括外层范围不变、纯函数成本分离、配置变化时缓存失效、返回对象隔离、两个PCD独立变化、联动总成本选择和候选预算。

### 搜索、剪枝与缓存

双PCD在0.05m网格上独立组合；先根据中心净空、径向间距、台柱保护层排除必然不合几何条件的网格点。每个规格/数量组合从剩余二维网格采用逐级包含的确定性采样，默认8对，另显式保留固定基线。可用`--anchor-pcd-samples`调整密度，或用`--anchor-full-pcd-grid`复核最终候选；绝不调用内层GA。

对抽样候选，锚栓违反度是完整联动违反度的下界；若下界已差于当前最优完整违反度，则跳过昂贵基础复核。相同外包尺寸的基础验算在本次内层调用中复用；完整搜索结果采用涵盖参数、配置、荷载和选择上下文的精确缓存键。缓存按进程独立、最多2048项，返回副本，防止调用者污染缓存。结果中的`anchor_solver_stats_main_process`仅为主进程统计，不能当作所有工作进程合计。

这属于有预算的离散近似搜索，并非完整网格或连续域的全局最优保证。PCD上下界或预算改变后应重新做收敛性分析。默认数量网格为60至160、步长4；每圈90根基线额外纳入，以便候选集合包含比较基线。

初始化：fixed读取新基线的真实外包直径；nested按允许规格的最小可能外包尺寸计算乐观存在性边界，避免旧固定锚笼尺寸误删可行基础。预筛选通过不等于整体可行。

### 同一成本口径

`基础成本 = (承台混凝土体积 + 桩身混凝土体积) × 混凝土单价 + (承台钢筋 + 桩身钢筋 + 原模型承台附加钢筋) × 钢筋单价`。

`锚栓成本 = 两圈锚栓数量 × 对应单价 + 法兰系数 × π × 两圈平均PCD × 法兰每米单价`。

`总成本 = 基础成本 + 锚栓成本`。fixed/nested共用同一实现；主目标保留“总成本/混凝土单价”的等效体积量级。罚值仅由外层约束处理使用，不进入真实成本字段。材料估算不含土方、安装、运输和税费。旧主目标漏计桩身混凝土，本次在两种模式中同时补齐，故不能直接对照历史旧目标值宣称成本下降。

兼容说明：独立旧分析脚本若不传`params["anchor"]`，仍可使用原`ANCHOR_CAGE`几何回退；新主优化流程始终显式传入候选。旧独立`optimize_anchor_cage.py`没有被替换。需复核新结果时使用`result.json`里的`best_params`，避免只解码22个基因而遗漏内层候选。

## 验证与比较

```powershell
python -m py_compile foundation_config.py anchor_cage_solver.py optimize_foundation_ga.py analyze_integrated_pile_foundation.py experiment_runner.py test_anchor_nested.py
python -m unittest test_anchor_nested -v
```

实测采用种子0和1、种群8、代数2、1/2进程。8次运行全部完成；单/双进程相同种子的基因、成本、锚栓及逐项约束完全一致。两种模式的整体可行运行率和最终种群可行率均为0。各成本仅是不可行候选的诊断数据。

校准后比较至少包含fixed 22变量GA与nested 22+4。保持种子、种群、代数、核数、算子、局部搜索、荷载和材料相同，报告实际外层评价次数、运行时间、可行运行率、最终种群可行率及仅可行方案成本分项。不要比较不可行解的“节省率”。

## 待施工图及荷载资料校准

当前公开资料研究基线：PCD_in=5.8m、PCD_out=6.3m、每圈90根、M56、总长4.0m。nested搜索M48/M56/M64/M72、每圈72～108根，PCD范围5.2～7.0m。M72价格、套管尺寸及全部规格的材料参数仍需供应商资料校准。

疲劳弯矩默认改为0.10M，并通过`--fatigue-moment-ratio`扫描0.05、0.10、0.15、0.20、0.30。该比例不是规范或厂家真实疲劳谱。也可在`ANCHOR_OPT["FATIGUE_DELTA_M_KNM"]`显式提供弯矩范围。35d、50MPa、套管与保护层、构造净距（含0.6m台柱/桩位避让）、瓶颈限制、荷载分项口径、锚栓/法兰价格及简化受力公式均为待校准研究参数。本模型不构成规范合规确认；预紧力、疲劳谱及循环数、锚板厚度、局部锚固区等仍需完整工程复核。

## 备份与审查

修改前四个文件位于`backups/anchor_22_plus_4_20260922_164340/`，附SHA-256清单。`_anchor_merge_review/*.merged.diff`为原项目到实际合并结果的可审查差异。保留了项目已有混合变量、epsilon、局部搜索及布局相关改动。
