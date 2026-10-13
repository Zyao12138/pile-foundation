import pathlib

p = pathlib.Path(r"C:\Users\18445\PyCharmMiscProject\pile foundation\run_multistart_optimization_and_verify.py")
text = p.read_text(encoding="utf-8")

old = '''    # rings 单独处理(list of dict)
    simple["geo"]["rings"] = [{"n": r["n"], "r": r["r"]} for r in params["geo"]["rings"]]'''

new = '''    # rings 单独处理(list of dict)
    simple["geo"]["rings"] = [{"n": r["n"], "r": r["r"]} for r in params["geo"]["rings"]]
    # 嵌套dict单独补上 (之前被类型过滤掉了)
    simple["rebar"]["ring_opt"] = params["rebar"]["ring_opt"]
    simple["rebar"]["top_ring_opt"] = params["rebar"]["top_ring_opt"]'''

assert old in text, "target block not found"
text = text.replace(old, new)
p.write_text(text, encoding="utf-8")
print("Fixed: ring_opt and top_ring_opt now preserved in JSON")
