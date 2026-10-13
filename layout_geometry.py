# -*- coding: utf-8 -*-
"""
锚栓笼与径向钢筋排布的统一几何检测模块。

核心功能：
    1. 统一钢筋直径单位。
    2. 同时考虑内外两圈锚栓。
    3. 根据钢筋线段的真实起止半径计算最小净距。
    4. 对发生碰撞的钢筋角度进行最小角度避让。
"""

import math

import numpy as np


def safe_float(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return float(value[0])
    return float(value)


def normalize_diameter_m(value):
    """同时兼容 0.022 m 和 22 mm 两种直径输入。"""
    raw = safe_float(value)
    return raw / 1000.0 if raw > 1.0 else raw


def normalize_anchor_config(ach_config):
    config = ach_config or {}
    pcd_in = safe_float(config.get("PCD_in", config.get("PCD", 4.8)))
    bolt_d_mm = safe_float(config.get("BOLT_D", config.get("Bolt_D_mm", 48.0)))
    sleeve_d_mm = safe_float(config.get("d_sleeve", config.get("d_sleeve_mm", bolt_d_mm + 17.0)))
    pcd_out = safe_float(
        config.get(
            "PCD_out",
            max(pcd_in + 0.4, config.get("D_outer", pcd_in + 0.4)),
        )
    )
    n_per_ring = int(
        round(
            safe_float(
                config.get(
                    "N_per_ring",
                    config.get("N_BOLTS", 180) / 2.0,
                )
            )
        )
    )
    clear_gap_m = safe_float(config.get("clear_gap_mm", 50.0)) / 1000.0
    return {
        "PCD_in": pcd_in,
        "PCD_out": pcd_out,
        "N_per_ring": max(1, n_per_ring),
        "d_sleeve_m": sleeve_d_mm / 1000.0,
        "bolt_d_m": bolt_d_mm / 1000.0,
        "clear_gap_m": clear_gap_m,
    }


def bolt_centers(ach_config):
    config = normalize_anchor_config(ach_config)
    angles = np.linspace(0.0, 2.0 * math.pi, config["N_per_ring"], endpoint=False)
    centers = []
    for radius in (config["PCD_in"] / 2.0, config["PCD_out"] / 2.0):
        for angle in angles:
            centers.append((
                radius * math.cos(angle),
                radius * math.sin(angle),
                config["d_sleeve_m"] / 2.0,
            ))
    return centers


def point_to_segment_distance(point, start, end):
    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1e-18:
        return float(np.linalg.norm(point - start))
    projection = float(np.dot(point - start, segment) / length_sq)
    projection = max(0.0, min(1.0, projection))
    closest = start + projection * segment
    return float(np.linalg.norm(point - closest))


def segment_clearance_metrics(angle, r_start, r_end, rebar_diameter, ach_config):
    """
    计算一根径向钢筋与所有锚栓套管的最小净距和碰撞数量。
    """
    rebar_radius = normalize_diameter_m(rebar_diameter) / 2.0
    start = (r_start * math.cos(angle), r_start * math.sin(angle))
    end = (r_end * math.cos(angle), r_end * math.sin(angle))

    min_clearance = float("inf")
    collisions = 0
    critical_radius = 0.0

    for x, y, sleeve_radius in bolt_centers(ach_config):
        distance = point_to_segment_distance((x, y), start, end)
        clearance = distance - sleeve_radius - rebar_radius
        if clearance < min_clearance:
            min_clearance = clearance
            critical_radius = math.hypot(x, y)
        if clearance < 0.0:
            collisions += 1

    if math.isinf(min_clearance):
        min_clearance = float("inf")

    return {
        "min_clearance_m": min_clearance,
        "collisions": collisions,
        "critical_radius_m": critical_radius,
    }


def get_nudged_angles(
    n_rebar,
    rebar_diameter,
    ach_config,
    offset_angle=0.0,
    r_start=0.0,
    r_end=None,
    max_iterations=6,
):
    """
    为径向钢筋生成满足净距要求的避让角度。

    r_start/r_end 表示该组钢筋的实际径向范围。
    """
    count = int(round(safe_float(n_rebar)))
    if count <= 0:
        return []

    config = normalize_anchor_config(ach_config)
    if r_end is None:
        r_end = max(config["PCD_out"] / 2.0 + 0.5, r_start + 0.1)

    initial_angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False) + offset_angle
    bolt_data = []
    for radius in (config["PCD_in"] / 2.0, config["PCD_out"] / 2.0):
        if r_start - 1e-9 <= radius <= r_end + 1e-9:
            angles = np.linspace(0.0, 2.0 * math.pi, config["N_per_ring"], endpoint=False)
            bolt_data.append((radius, angles))

    rebar_radius = normalize_diameter_m(rebar_diameter) / 2.0
    required_radius = config["d_sleeve_m"] / 2.0 + rebar_radius + config["clear_gap_m"]

    nudged = []
    for initial in initial_angles:
        current = float(initial % (2.0 * math.pi))
        for _ in range(int(max_iterations)):
            moved = False
            for radius, bolt_angles in bolt_data:
                if required_radius >= radius:
                    continue
                safe_angle = math.asin(min(1.0, required_radius / radius))
                diffs = np.abs(bolt_angles - current)
                diffs = np.minimum(diffs, 2.0 * math.pi - diffs)
                nearest_index = int(np.argmin(diffs))
                if diffs[nearest_index] >= safe_angle:
                    continue

                bolt_angle = float(bolt_angles[nearest_index])
                delta = (current - bolt_angle + math.pi) % (2.0 * math.pi) - math.pi
                current = bolt_angle + safe_angle if delta >= 0.0 else bolt_angle - safe_angle
                current %= 2.0 * math.pi
                moved = True
            if not moved:
                break
        nudged.append(current)

    return nudged


def evaluate_rebar_groups(groups, ach_config):
    """
    汇总多组钢筋的最小净距和碰撞情况。

    groups: [
        {"name": "n1", "count": 160, "diameter": 22, "r_start": 1.5, "r_end": 10.3},
        ...
    ]
    """
    results = {}
    global_min = float("inf")
    total_collisions = 0

    for group in groups:
        angles = get_nudged_angles(
            group["count"],
            group["diameter"],
            ach_config,
            offset_angle=group.get("offset_angle", 0.0),
            r_start=group["r_start"],
            r_end=group["r_end"],
        )
        group_min = float("inf")
        group_collisions = 0
        for angle in angles:
            metrics = segment_clearance_metrics(
                angle,
                group["r_start"],
                group["r_end"],
                group["diameter"],
                ach_config,
            )
            group_min = min(group_min, metrics["min_clearance_m"])
            group_collisions += metrics["collisions"]

        results[group["name"]] = {
            "angles": angles,
            "min_clearance_m": group_min,
            "collisions": group_collisions,
        }
        global_min = min(global_min, group_min)
        total_collisions += group_collisions

    return {
        "groups": results,
        "minimum_clearance_m": global_min,
        "total_collisions": total_collisions,
    }
