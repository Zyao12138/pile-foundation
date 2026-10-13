# -*- coding: utf-8 -*-
"""
Abaqus 后处理脚本：自动提取桩顶 P-S 曲线数据 (力控制修正版)
"""
from odbAccess import openOdb

# 1. 打开结果文件
odb_name = 'Run_Pile_Soil_Interaction_ULS.odb'
odb = openOdb(path=odb_name)
step = odb.steps['ULS_Load_Step']

# 你在 JSON 中设定的最大轴压 ULS 荷载 (kN)
MAX_LOAD_KN = 2704.69

target_node = None
for node in odb.rootAssembly.nodes:
    if abs(node.coordinates[0] - 0.0) < 1e-5 and \
            abs(node.coordinates[1] - 0.0) < 1e-5 and \
            abs(node.coordinates[2] - 0.0) < 1e-5:
        target_node = node
        break

if target_node is None:
    print("Error: Could not find the Reference Point at (0,0,0).")
else:
    print("Successfully found Reference Point. Extracting Data...")
    print("--------------------------------------------------")
    print(" Step_Time | Settlement(mm) | Top_Load(kN)")
    print("--------------------------------------------------")

    for frame in step.frames:
        time = frame.frameValue

        # 提取当前帧的位移 (U3)
        u_field = frame.fieldOutputs['U']
        try:
            u3_val = u_field.getSubset(region=target_node).values[0].data[2]
            settlement_mm = abs(u3_val) * 1000.0

            # 【核心修复】：直接利用线性比例计算当前力
            load_kn = MAX_LOAD_KN * time

            print(" {:9.4f} | {:14.4f} | {:12.2f}".format(time, settlement_mm, load_kn))
        except IndexError:
            pass

    print("--------------------------------------------------")
    print("Data Extraction Completed!")

odb.close()