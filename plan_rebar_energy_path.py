# -*- coding: utf-8 -*-
"""
独立模块：基于能量最小化（全局路径规划）的钢筋平滑避障引擎
算法：非线性无约束优化 (SciPy L-BFGS-B) / 弹性带模型
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.patches as patches


class EnergyMinimizationRebar:
    def __init__(self, start_pt, end_pt, obstacle_center, r_obstacle, num_points=20):
        self.start_pt = np.array(start_pt)
        self.end_pt = np.array(end_pt)
        self.obs = np.array(obstacle_center)
        self.r_obs = r_obstacle
        self.n_pts = num_points

        # 初始化钢筋为一条直线上的离散点
        self.initial_points = np.linspace(self.start_pt, self.end_pt, self.n_pts)

    def energy_function(self, coords_1d, weight_bend=1.0, weight_repel=5.0):
        """
        核心目标函数：计算整根钢筋的“总能量”
        coords_1d: 优化器传入的一维数组 [x1, y1, x2, y2...] (剔除了首尾固定点)
        """
        # 将一维数组重组为二维坐标点阵
        pts = np.vstack([self.start_pt, coords_1d.reshape(-1, 2), self.end_pt])

        # 1. 弯曲应变能 (Internal Energy - 曲率惩罚)
        # 利用二阶差分近似二阶导数： |P(i-1) - 2P(i) + P(i+1)|^2
        bending_energy = 0.0
        for i in range(1, len(pts) - 1):
            bend_vec = pts[i - 1] - 2 * pts[i] + pts[i + 1]
            bending_energy += np.sum(bend_vec ** 2)

        # 2. 障碍物斥力能 (External Energy - 避障惩罚)
        repulsion_energy = 0.0
        for pt in pts:
            dist = np.linalg.norm(pt - self.obs)
            # 如果进入障碍物内部，施加极其巨大的惩罚
            if dist < self.r_obs:
                repulsion_energy += 1e6 * (self.r_obs - dist)
            else:
                # 距离越近，斥力越大 (呈反比)
                repulsion_energy += 1.0 / (dist - self.r_obs + 1e-3) ** 2

        return weight_bend * bending_energy + weight_repel * repulsion_energy

    def optimize_path(self):
        """执行最优化求解"""
        # 提取中间的可动控制点作为初始猜测值 (展平为1维)
        guess_pts = self.initial_points[1:-1].flatten()

        # 运行最优化算法 (L-BFGS-B 是一种高效的拟牛顿法)
        # 这里算法会自动微调点的位置，直到 energy_function 达到极小值
        res = minimize(self.energy_function, guess_pts, method='L-BFGS-B',
                       options={'maxiter': 1000, 'disp': False})

        optimized_pts = np.vstack([self.start_pt, res.x.reshape(-1, 2), self.end_pt])
        return optimized_pts

    def visualize(self):
        """可视化结果：对比局部直角避让与全局平滑避让"""
        opt_path = self.optimize_path()

        fig, ax = plt.subplots(figsize=(8, 8))

        # 画障碍物 (如锚栓或桩头)
        circle = patches.Circle(self.obs, self.r_obs, color='red', alpha=0.5, label='Obstacle (Bolt/Pile)')
        ax.add_patch(circle)

        # 画原始直线 (会穿模)
        ax.plot(self.initial_points[:, 0], self.initial_points[:, 1], 'k--', alpha=0.4,
                label='Straight Line (Collision)')

        # 画局部贪心折线 (模拟你之前的 Nudging，直接在障碍物处折弯)
        naive_mid_pt = self.obs + np.array([-self.r_obs - 0.02, 0])  # 强制向左偏移
        ax.plot([self.start_pt[0], naive_mid_pt[0], self.end_pt[0]],
                [self.start_pt[1], naive_mid_pt[1], self.end_pt[1]],
                'orange', linewidth=2, linestyle='-.', label='Local Nudging (High Strain)')

        # 画全局能量最小化曲线
        ax.plot(opt_path[:, 0], opt_path[:, 1], 'blue', linewidth=3, label='Energy Minimized Spline (Smooth)')
        ax.scatter(opt_path[:, 0], opt_path[:, 1], c='blue', s=10)  # 显示控制点

        ax.set_aspect('equal')
        plt.legend()
        plt.title("Heuristic Global Path Planning (Energy Minimization)")
        plt.grid(True, linestyle=':')
        plt.show()


# ================= 测试入口 =================
if __name__ == "__main__":
    # 模拟一根钢筋从 (0,0) 走向 (0, 5)
    # 锚栓在 (0, 2.5) 的位置，阻挡了去路
    router = EnergyMinimizationRebar(
        start_pt=(0.0, 0.0),
        end_pt=(0.0, 5.0),
        obstacle_center=(0.0, 2.5),
        r_obstacle=0.3  # 障碍物半径+安全间距
    )
    router.visualize()