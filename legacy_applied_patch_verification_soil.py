import pathlib, copy

p = pathlib.Path(r"C:\Users\18445\PyCharmMiscProject\pile foundation\verify_design_constraints.py")
text = p.read_text(encoding="utf-8")

# ---- 1. 在 soil_layers 定义后构造 eff_soil_layers ----
old_soil = '''    soil_layers = data['soil_layers']
    piles_coords = data['piles_coords']
    n_piles = len(piles_coords)'''

new_soil = '''    soil_layers = data['soil_layers']
    piles_coords = data['piles_coords']
    n_piles = len(piles_coords)

    # 构造有效土层: 剔除承台埋深H以上的土, 与优化器口径一致
    H_cap_actual = data['cap_geo'].get('H', 3.6)
    eff_soil_layers = []
    d_acc = 0.0
    for lyr in soil_layers:
        if d_acc + lyr.thickness <= H_cap_actual:
            d_acc += lyr.thickness
            continue
        new_lyr = copy.copy(lyr)
        if d_acc < H_cap_actual:
            new_lyr.thickness = (d_acc + lyr.thickness) - H_cap_actual
        eff_soil_layers.append(new_lyr)
        d_acc += lyr.thickness'''

assert old_soil in text, "soil block not found"
text = text.replace(old_soil, new_soil)

# ---- 2. 替换桩基承载力相关验算的 soil_layers -> eff_soil_layers ----
replacements = [
    # P-1 抗压
    ("pile_analyzer.calculate_compression_capacity(data['pile_geo']['L'], soil_layers)",
     "pile_analyzer.calculate_compression_capacity(data['pile_geo']['L'], eff_soil_layers)"),
    # P-2 抗拔
    ("pile_analyzer.calculate_uplift_capacity(data['pile_geo']['L'], soil_layers, piles_coords)",
     "pile_analyzer.calculate_uplift_capacity(data['pile_geo']['L'], eff_soil_layers, piles_coords)"),
    # P-3 水平
    ("pile_analyzer.calculate_horizontal_capacity(data['pile_geo']['L'], soil_layers)",
     "pile_analyzer.calculate_horizontal_capacity(data['pile_geo']['L'], eff_soil_layers)"),
    # P-6 压屈
    ("pile_analyzer.check_buckling_stability(max_N_comp_uls, data['pile_geo']['L'], soil_layers)",
     "pile_analyzer.check_buckling_stability(max_N_comp_uls, data['pile_geo']['L'], eff_soil_layers)"),
    # P-11 负摩阻
    ("l.thickness for l in soil_layers if getattr(l, 'is_compressible', False)), soil_layers, piles_coords)",
     "l.thickness for l in eff_soil_layers if getattr(l, 'is_compressible', False)), eff_soil_layers, piles_coords)"),
    # FEM 水平内力 (两处)
    ("1.0, 0, data['pile_geo']['L'], soil_layers, piles_coords)",
     "1.0, 0, data['pile_geo']['L'], eff_soil_layers, piles_coords)"),
]

for old, new in replacements:
    if old in text:
        text = text.replace(old, new)
        print(f"Replaced: {old[:50]}...")
    else:
        print(f"[WARN] Not found: {old[:50]}...")

# ---- 3. P-10 和 P-12 传入实际 H_cap ----
old_p10 = "pile_analyzer.check_weak_underlying_layer(F_z_sls, 0.0, data['pile_geo']['L'], soil_layers, piles_coords)"
new_p10 = "pile_analyzer.check_weak_underlying_layer(F_z_sls, 0.0, data['pile_geo']['L'], soil_layers, piles_coords, H_cap=H_cap_actual)"
if old_p10 in text:
    text = text.replace(old_p10, new_p10)
    print("Fixed P-10 H_cap")
else:
    print("[WARN] P-10 line not found")

old_p12 = "pile_analyzer.check_settlement_and_tilt(F_z_sls, M_xy_sls, data['pile_geo']['L'], soil_layers,\n                                                        piles_coords)"
new_p12 = "pile_analyzer.check_settlement_and_tilt(F_z_sls, M_xy_sls, data['pile_geo']['L'], soil_layers,\n                                                        piles_coords, H_cap=H_cap_actual)"
if old_p12 in text:
    text = text.replace(old_p12, new_p12)
    print("Fixed P-12 H_cap")
else:
    print("[WARN] P-12 line not found")

p.write_text(text, encoding="utf-8")
print("\nVerification.py fixed: eff_soil_layers + actual H_cap")
