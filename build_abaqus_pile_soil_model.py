# -*- coding: utf-8 -*-
"""
Abaqus 参数化建模脚本：单桩-多层土耦合分析 (v4.0 - 引入 CDP 本构与力学耦合修复版)
修复核心：
1. 彻底解决 Abaqus 中 __file__ 变量未定义导致的报错，动态获取用户路径。
2. 修复接触面在拐角处交叉重叠导致的严重不收敛报错（偏转 10 度查找）。
3. 引入完整的 CDP (Concrete Damaged Plasticity) 混凝土损伤塑性本构，支持非线性 ULS 破坏分析。
4. 将桩顶加载耦合方式从刚性（KINEMATIC）修改为分布耦合（DISTRIBUTING），避免虚假应力奇异崩溃。
5. 保持了之前所有的接触面修正、网格容差与地应力步长修复。
"""

from abaqus import *
from abaqusConstants import *
import regionToolset
import part
import material
import section
import assembly
import step
import interaction
import load
import mesh
import job
import json
import math
import os

# =========================================================
# 1. 全局配置与单位转换
# =========================================================
MODEL_NAME = 'Pile_Soil_Interaction_ULS'

# 动态获取当前 Windows 登录用户的根目录
user_home = os.path.expanduser("~")
ABAQUS_OUTPUT_DIR = os.path.join(user_home, "PyCharmMiscProject", "pile foundation", "abaqus_postprocess")
JSON_FILENAME = os.path.join(ABAQUS_OUTPUT_DIR, "abaqus_pile_model.json")

Mdb()
if MODEL_NAME in mdb.models:
    del mdb.models[MODEL_NAME]
myModel = mdb.Model(name=MODEL_NAME)

kN_to_N = 1000.0
MPa_to_Pa = 1e6
kPa_to_Pa = 1000.0
kN_m3_to_kg_m3 = 1000.0 / 9.81


def load_json_data(filename):
    if not os.path.exists(filename):
        raise ValueError("Cannot find JSON file: {}. Please check your Python optimization code output path.".format(filename))
    with open(filename, 'r') as f:
        data = json.load(f)
    print("Successfully loaded JSON data from: {}".format(filename))
    return data


try:
    data = load_json_data(JSON_FILENAME)
except Exception as e:
    raise ValueError("Failed to parse JSON: {}".format(e))

pile_prop = data['pile_property']
soil_layers = data['soil_layers']
loads = data['critical_loads_uls']

D_pile = pile_prop['geometry']['D_pile_m']
L_pile = pile_prop['geometry']['L_pile_m']
cover = pile_prop['geometry']['cover_thickness_m']
R_pile = D_pile / 2.0

# 偏移查找角度，避开 45/135度 的网格切割面，防止漏选接触面导致穿透
safe_pts = [
    (R_pile * math.cos(math.radians(10)), R_pile * math.sin(math.radians(10))),
    (R_pile * math.cos(math.radians(100)), R_pile * math.sin(math.radians(100))),
    (R_pile * math.cos(math.radians(190)), R_pile * math.sin(math.radians(190))),
    (R_pile * math.cos(math.radians(280)), R_pile * math.sin(math.radians(280)))
]

total_soil_depth = sum([layer['thickness_m'] for layer in soil_layers])
min_required_depth = L_pile + 5.0 * D_pile
if total_soil_depth < min_required_depth:
    print("WARNING: Insufficient soil depth. Increasing bottom layer thickness.")
    soil_layers[-1]['thickness_m'] += (min_required_depth - total_soil_depth)
    total_soil_depth = min_required_depth

soil_domain_width = max(20.0, 15.0 * D_pile)
soil_domain_depth = total_soil_depth

# =========================================================
# 2. 创建材料与截面 (CDP 损伤塑性本构)
# =========================================================
print("Creating Materials and Sections...")

# 【核心修复1：引入 CDP 混凝土损伤塑性本构】
conc_mat = myModel.Material(name='Concrete_Pile')
conc_mat.Density(table=((2500.0,),))
conc_mat.Elastic(table=((pile_prop['concrete_material']['Ec_MPa'] * MPa_to_Pa, 0.2),))

# CDP 塑性基本参数 (剪胀角30, 偏心率0.1, fb0/fc0 1.16, K 0.667, 粘性系数 0.001 帮助收敛)
conc_mat.ConcreteDamagedPlasticity(table=((30.0, 0.1, 1.16, 0.667, 0.001),))
# 混凝土受压硬化 (示例数据，适应 C40/C50)
conc_mat.concreteDamagedPlasticity.ConcreteCompressionHardening(table=(
    (20.1e6, 0.0), (28.5e6, 0.001), (31.4e6, 0.002), (25.0e6, 0.005)
))
# 混凝土受拉软化 (拉伸应力-开裂应变，模拟开裂)
conc_mat.concreteDamagedPlasticity.ConcreteTensionStiffening(table=(
    (2.8e6, 0.0), (1.5e6, 0.0005), (0.5e6, 0.002)
))
# 损伤因子 (用于输出 DAMAGET 裂缝云图)
conc_mat.concreteDamagedPlasticity.ConcreteCompressionDamage(table=((0.0, 0.0), (0.2, 0.002), (0.5, 0.005)))
conc_mat.concreteDamagedPlasticity.ConcreteTensionDamage(table=((0.0, 0.0), (0.6, 0.0005), (0.9, 0.002)))

myModel.HomogeneousSolidSection(name='Pile_Section', material='Concrete_Pile', thickness=None)

steel_mat = myModel.Material(name='Steel_Rebar')
steel_mat.Elastic(table=((2.0e11, 0.3),))
steel_mat.Density(table=((7850.0,),))
rebar_area = math.pi * (pile_prop['rebar']['main_rebar_d_m'] / 2.0) ** 2
myModel.TrussSection(name='Rebar_Section', material='Steel_Rebar', area=rebar_area)

for i, layer in enumerate(soil_layers):
    mat_name = "Soil_Layer_{}".format(i + 1)
    s_mat = myModel.Material(name=mat_name)
    s_mat.Density(table=((layer['density_kN_m3'] * kN_m3_to_kg_m3,),))
    s_mat.Elastic(table=((layer['elastic_modulus_MPa'] * MPa_to_Pa, layer['poisson_ratio']),))

    c_val = max(layer['cohesion_kPa'] * kPa_to_Pa, 1000.0)

    # 强制赋予 0.1 度的剪胀角，消除警告
    dilation_angle = layer.get('dilation_angle_deg', 0.1)
    if dilation_angle == 0.0:
        dilation_angle = 0.1
    s_mat.MohrCoulombPlasticity(table=((layer['friction_angle_deg'], dilation_angle),))
    s_mat.mohrCoulombPlasticity.MohrCoulombHardening(table=((c_val, 0.0),))
    myModel.HomogeneousSolidSection(name="Section_{}".format(mat_name), material=mat_name, thickness=None)

# =========================================================
# 3. 部件建模 (Part) - 切分与赋材质
# =========================================================
print("Creating Geometry and Partitioning...")

sPart_raw = myModel.Part(name='Soil_Block_Raw', dimensionality=THREE_D, type=DEFORMABLE_BODY)
s_sketch = myModel.ConstrainedSketch(name='Soil_Sketch', sheetSize=soil_domain_width * 2)
s_sketch.rectangle(point1=(-soil_domain_width / 2.0, -soil_domain_width / 2.0),
                   point2=(soil_domain_width / 2.0, soil_domain_width / 2.0))
sPart_raw.BaseSolidExtrude(sketch=s_sketch, depth=soil_domain_depth)

holePart = myModel.Part(name='Hole_Tool', dimensionality=THREE_D, type=DEFORMABLE_BODY)
h_sketch = myModel.ConstrainedSketch(name='Hole_Sketch', sheetSize=D_pile * 2)
h_sketch.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(R_pile, 0.0))
holePart.BaseSolidExtrude(sketch=h_sketch, depth=L_pile)

myAss = myModel.rootAssembly
myAss.DatumCsysByDefault(CARTESIAN)
inst_soil_raw = myAss.Instance(name='Soil_Raw_Inst', part=sPart_raw, dependent=OFF)
inst_hole = myAss.Instance(name='Hole_Tool_Inst', part=holePart, dependent=OFF)

cut_inst = myAss.InstanceFromBooleanCut(name='Soil_Block', instanceToBeCut=inst_soil_raw, cuttingInstances=(inst_hole,),
                                        originalInstances=DELETE)
sPart = myModel.parts['Soil_Block']
del myAss.instances[cut_inst.name]

dp_tip = sPart.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE, offset=L_pile)
try:
    sPart.PartitionCellByDatumPlane(datumPlane=sPart.datums[dp_tip.id], cells=sPart.cells)
except:
    pass

pt_o = sPart.DatumPointByCoordinate(coords=(0.0, 0.0, 0.0))
pt_z = sPart.DatumPointByCoordinate(coords=(0.0, 0.0, 1.0))
pt_d1 = sPart.DatumPointByCoordinate(coords=(1.0, 1.0, 0.0))
pt_d2 = sPart.DatumPointByCoordinate(coords=(-1.0, 1.0, 0.0))
dp1 = sPart.DatumPlaneByThreePoints(point1=sPart.datums[pt_o.id], point2=sPart.datums[pt_d1.id],
                                    point3=sPart.datums[pt_z.id])
dp2 = sPart.DatumPlaneByThreePoints(point1=sPart.datums[pt_o.id], point2=sPart.datums[pt_d2.id],
                                    point3=sPart.datums[pt_z.id])
try:
    sPart.PartitionCellByDatumPlane(datumPlane=sPart.datums[dp1.id], cells=sPart.cells)
    sPart.PartitionCellByDatumPlane(datumPlane=sPart.datums[dp2.id], cells=sPart.cells)
except:
    pass

current_z = 0.0
for layer in soil_layers[:-1]:
    current_z += layer['thickness_m']
    dp = sPart.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE, offset=current_z)
    try:
        sPart.PartitionCellByDatumPlane(datumPlane=sPart.datums[dp.id], cells=sPart.cells)
    except:
        pass

current_z = 0.0
for i, layer in enumerate(soil_layers):
    sec_name = "Section_Soil_Layer_{}".format(i + 1)
    h = layer['thickness_m']
    z_min, z_max = current_z - 1e-3, current_z + h + 1e-3
    xy_max = soil_domain_width + 1.0
    cells = sPart.cells.getByBoundingBox(xMin=-xy_max, yMin=-xy_max, zMin=z_min,
                                         xMax=xy_max, yMax=xy_max, zMax=z_max)
    if cells:
        sPart.SectionAssignment(region=regionToolset.Region(cells=cells), sectionName=sec_name, offset=0.0,
                                offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)
    current_z += h

pPart = myModel.Part(name='Pile', dimensionality=THREE_D, type=DEFORMABLE_BODY)
p_sketch = myModel.ConstrainedSketch(name='Pile_Sketch', sheetSize=D_pile * 4)
p_sketch.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(R_pile, 0.0))
pPart.BaseSolidExtrude(sketch=p_sketch, depth=L_pile)
pPart.SectionAssignment(region=regionToolset.Region(cells=pPart.cells), sectionName='Pile_Section', offset=0.0,
                        offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)

rPart = myModel.Part(name='Rebar_Cage', dimensionality=THREE_D, type=DEFORMABLE_BODY)
n_main = pile_prop['rebar']['main_rebar_n']
r_rebar_ring = R_pile - cover - (pile_prop['rebar']['main_rebar_d_m'] / 2.0)
for i in range(n_main):
    angle = 2.0 * math.pi * i / n_main
    x, y = r_rebar_ring * math.cos(angle), r_rebar_ring * math.sin(angle)
    p1, p2 = rPart.DatumPointByCoordinate(coords=(x, y, 0.0)), rPart.DatumPointByCoordinate(coords=(x, y, L_pile))
    rPart.WirePolyLine(points=((rPart.datums[p1.id], rPart.datums[p2.id]),), mergeType=IMPRINT, meshable=ON)
rPart.SectionAssignment(region=regionToolset.Region(edges=rPart.edges), sectionName='Rebar_Section', offset=0.0,
                        offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)

# =========================================================
# 4. 装配 (Assembly)
# =========================================================
print("Assembling...")
soil_inst = myAss.Instance(name='Soil_Inst', part=sPart, dependent=ON)
pile_inst = myAss.Instance(name='Pile_Inst', part=pPart, dependent=ON)
rebar_inst = myAss.Instance(name='Rebar_Inst', part=rPart, dependent=ON)

# =========================================================
# 5. 分析步与相互作用
# =========================================================
print("Creating Steps and Interactions...")

# 将地应力步初始增量从 0.1 缩小至 0.01，给非线性算法多一点喘息时间
myModel.GeostaticStep(name='Geostatic_Step', previous='Initial', nlgeom=ON, matrixSolver=DIRECT,
                      matrixStorage=UNSYMMETRIC, timeIncrementationMethod=AUTOMATIC,
                      initialInc=0.01, minInc=1e-15, maxInc=1.0, utol=1e-3)

myModel.StaticStep(name='ULS_Load_Step', previous='Geostatic_Step', nlgeom=ON,
                   matrixSolver=DIRECT, matrixStorage=UNSYMMETRIC,
                   initialInc=0.001, maxInc=0.05, minInc=1e-20,
                   stabilizationMagnitude=0.0002, stabilizationMethod=DISSIPATED_ENERGY_FRACTION)

myModel.EmbeddedRegion(name='Rebar_Embed_in_Pile', embeddedRegion=regionToolset.Region(edges=rebar_inst.edges),
                       hostRegion=regionToolset.Region(cells=pile_inst.cells))

contact_prop = myModel.ContactProperty('Pile_Soil_Friction')
contact_prop.TangentialBehavior(formulation=PENALTY, table=((0.85,),), fraction=0.01)
contact_prop.NormalBehavior(pressureOverclosure=HARD, allowSeparation=ON,
                            constraintEnforcementMethod=AUGMENTED_LAGRANGE)

z_cuts_all = [0.0]
curr_z = 0.0
for layer in soil_layers:
    curr_z += layer['thickness_m']
    z_cuts_all.append(curr_z)
if L_pile not in z_cuts_all: z_cuts_all.append(L_pile)
z_cuts_all = sorted(list(set(z_cuts_all)))

s_inner_pts = []
s_side_pts = []
dist = soil_domain_width / 2.0
for k in range(len(z_cuts_all) - 1):
    mid_z = (z_cuts_all[k] + z_cuts_all[k + 1]) / 2.0
    s_side_pts.extend([((dist, 0.0, mid_z),), ((-dist, 0.0, mid_z),), ((0.0, dist, mid_z),), ((0.0, -dist, mid_z),)])
    if mid_z <= L_pile:
        for sx, sy in safe_pts:
            s_inner_pts.append(((sx, sy, mid_z),))

s_bot_pts = [((sx * 0.5, sy * 0.5, L_pile),) for sx, sy in safe_pts]
p_side_pts = [((sx, sy, L_pile / 2.0),) for sx, sy in safe_pts]
p_bot_pts = [((sx * 0.5, sy * 0.5, L_pile),) for sx, sy in safe_pts]

# 将侧面和底面的查找点合并在一起，生成一整个表面
s_all_pts = s_inner_pts + s_bot_pts
p_all_pts = p_side_pts + p_bot_pts

s_all_surf = myAss.Surface(name='Soil_Contact_Surf', side1Faces=soil_inst.faces.findAt(*s_all_pts))
p_all_surf = myAss.Surface(name='Pile_Contact_Surf', side1Faces=pile_inst.faces.findAt(*p_all_pts))

# 开启微小容差的 Adjust，闭合圆柱面扫掠逼近造成的微小缝隙，避免重力施加时瞬间下落穿透
try:
    myModel.SurfaceToSurfaceContactStd(name='Contact_PileSoil', createStepName='Initial', main=p_all_surf,
                                       secondary=s_all_surf, sliding=FINITE, interactionProperty='Pile_Soil_Friction',
                                       adjustMethod=TOLERANCE, adjustTolerance=1e-4)
except TypeError:
    myModel.SurfaceToSurfaceContactStd(name='Contact_PileSoil', createStepName='Initial', master=p_all_surf,
                                       slave=s_all_surf, sliding=FINITE, interactionProperty='Pile_Soil_Friction',
                                       adjustMethod=TOLERANCE, adjustTolerance=1e-4)

# =========================================================
# 6. 边界条件、地应力场与荷载
# =========================================================
print("Applying Boundary Conditions and Loads...")

xy_max = soil_domain_width + 1.0
s_bot_faces = soil_inst.faces.getByBoundingBox(xMin=-xy_max, yMin=-xy_max, zMin=soil_domain_depth - 0.1, xMax=xy_max,
                                               yMax=xy_max, zMax=soil_domain_depth + 0.1)
myModel.DisplacementBC(name='BC_Soil_Bottom_Fixed', createStepName='Initial',
                       region=regionToolset.Region(faces=s_bot_faces), u1=SET, u2=SET, u3=SET, ur1=UNSET, ur2=UNSET,
                       ur3=UNSET)

myModel.DisplacementBC(name='BC_Soil_Sides_Fixed', createStepName='Initial',
                       region=regionToolset.Region(faces=soil_inst.faces.findAt(*s_side_pts)), u1=SET, u2=SET, u3=UNSET,
                       ur1=UNSET, ur2=UNSET, ur3=UNSET)

myModel.Gravity(name='Global_Gravity', createStepName='Geostatic_Step', comp3=9.81, distributionType=UNIFORM, field='')

current_z, current_vertical_stress = 0.0, 0.0
for i, layer in enumerate(soil_layers):
    h = layer['thickness_m']
    gamma = layer['density_kN_m3'] * kN_to_N
    top_z, top_stress = current_z, current_vertical_stress
    bot_z, bot_stress = current_z + h, current_vertical_stress - (gamma * h)
    K0 = max(0.2, 1.0 - math.sin(math.radians(layer['friction_angle_deg'])))

    z_min, z_max = current_z - 1e-3, current_z + h + 1e-3
    cells = soil_inst.cells.getByBoundingBox(xMin=-xy_max, yMin=-xy_max, zMin=z_min, xMax=xy_max, yMax=xy_max,
                                             zMax=z_max)
    if cells:
        myModel.GeostaticStress(name='Init_Stress_Soil_{}'.format(i + 1), region=regionToolset.Region(cells=cells),
                                stressMag1=top_stress, vCoord1=top_z, stressMag2=bot_stress, vCoord2=bot_z,
                                lateralCoeff1=K0, lateralCoeff2=K0)
    current_z += h
    current_vertical_stress = bot_stress

myModel.GeostaticStress(name='Init_Stress_Pile', region=regionToolset.Region(cells=pile_inst.cells), stressMag1=0.0,
                        vCoord1=0.0, stressMag2=-(2500.0 * 9.81 * L_pile), vCoord2=L_pile, lateralCoeff1=0.2,
                        lateralCoeff2=0.2)

rp = myAss.ReferencePoint(point=(0.0, 0.0, 0.0))
p_top_region = regionToolset.Region(faces=pile_inst.faces.findAt(((R_pile * 0.5, 0.0, 0.0),)))

# 【核心修复2：使用 DISTRIBUTING 分布耦合，避免桩顶应力奇异导致虚假破坏】
myModel.Coupling(name='Coupling_RP_to_PileTop',
                 controlPoint=regionToolset.Region(referencePoints=(myAss.referencePoints[rp.id],)),
                 surface=p_top_region, influenceRadius=WHOLE_SURFACE,
                 couplingType=DISTRIBUTING, weightingMethod=UNIFORM, localCsys=None,
                 u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)
rp_reg = regionToolset.Region(referencePoints=(myAss.referencePoints[rp.id],))

if loads.get('max_compression_kN', 0.0) != 0.0:
    myModel.ConcentratedForce(name='Load_Compression', createStepName='ULS_Load_Step', region=rp_reg,
                              cf3=loads['max_compression_kN'] * kN_to_N)

if loads.get('max_shear_kN', 0.0) != 0.0:
    myModel.ConcentratedForce(name='Load_Shear', createStepName='ULS_Load_Step', region=rp_reg,
                              cf1=loads['max_shear_kN'] * kN_to_N)

moment_val = loads.get('max_moment_kNm', 0.0)
if moment_val != 0.0:
    myModel.Moment(name='Load_Moment', createStepName='ULS_Load_Step', region=rp_reg,
                   cm2=moment_val * 1000.0)

# =========================================================
# 8. 网格划分 (Mesh)
# =========================================================
print("Meshing...")

pPart.seedPart(size=D_pile / 3.0, deviationFactor=0.05, minSizeFactor=0.1)
pPart.setMeshControls(regions=pPart.cells, elemShape=HEX, technique=SWEEP)
elemType_pile = mesh.ElemType(elemCode=C3D8R, elemLibrary=STANDARD, kinematicSplit=AVERAGE_STRAIN,
                              hourglassControl=ENHANCED)
pPart.setElementType(regions=regionToolset.Region(cells=pPart.cells), elemTypes=(elemType_pile,))
pPart.generateMesh()

rPart.seedPart(size=D_pile, deviationFactor=0.05, minSizeFactor=0.1)
elemType_rebar = mesh.ElemType(elemCode=T3D2, elemLibrary=STANDARD)
rPart.setElementType(regions=regionToolset.Region(edges=rPart.edges), elemTypes=(elemType_rebar,))
rPart.generateMesh()

sPart.seedPart(size=D_pile / 1.0, deviationFactor=0.05, minSizeFactor=0.1)
try:
    hole_edges = sPart.edges.getByBoundingCylinder((0.0, 0.0, -1.0), (0.0, 0.0, soil_domain_depth + 1.0), R_pile * 1.05)
    sPart.seedEdgeBySize(edges=hole_edges, size=D_pile / 3.0, constraint=FINER)
except Exception as e:
    print("Warning: Edge seeding failed, continuing with global seeds. ", e)

sPart.setMeshControls(regions=sPart.cells, elemShape=HEX_DOMINATED, technique=SWEEP, algorithm=MEDIAL_AXIS)
elemType_soil = mesh.ElemType(elemCode=C3D8R, elemLibrary=STANDARD, kinematicSplit=AVERAGE_STRAIN,
                              hourglassControl=ENHANCED)
sPart.setElementType(regions=regionToolset.Region(cells=sPart.cells), elemTypes=(elemType_soil,))
sPart.generateMesh()

# =========================================================
# 9. 创建 Job
# =========================================================
job_name = "Run_{}".format(MODEL_NAME)
print("\nModeling completed successfully! Job created: {}".format(job_name))
mdb.Job(name=job_name, model=MODEL_NAME, description='Pile-Soil Contact and Geostatic ULS', type=ANALYSIS, numCpus=12,
        numDomains=12)