#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 1×1 平面板（Nx×Ny 可细化），左边完全固定，
板面承受沿 +x 方向的均布面力 qx (N/m²)。
输出文件无注释，满足：
  Cantilever_Beam_T3 风格 → NUMNP, NUMEG, NLCASE, MODEX...
"""

def write_uniform_plate_T3(
        filename="plate_uniform_x_T3.txt",
        Nx=1, Ny=1,            # 网格细分
        L=1.0, H=1.0,          # 板长、板高 (m)
        qx=1000.0,             # 均布面力 +x (N/m²)
        E=2100.0, nu=0.0, t=1.0):   # 材料 & 厚度
    dx, dy = L / Nx, H / Ny

    # ---------- 节点 ----------
    nodes = []                              # (x, y, bx, by, bz)
    for ix in range(Nx + 1):
        x = ix * dx
        for iy in range(Ny + 1):
            y = iy * dy
            bx = by = 1 if ix == 0 else 0    # 左边固定
            nodes.append((x, y, bx, by, 1))  # bz 恒 1（平面问题）

    NUMNP  = len(nodes)
    NUMEG  = 2 * Nx * Ny                     # 每矩形 2 个三角形
    NLCASE = 1
    MODEX  = 1

    # ---------- 元素连接 ----------
    elems = []
    for ix in range(Nx):
        for iy in range(Ny):
            n1 = ix * (Ny + 1) + iy + 1
            n2 = n1 + (Ny + 1)
            n3 = n1 + 1
            n4 = n2 + 1
            elems += [(n1, n3, n2), (n3, n4, n2)]

    # ---------- 等效节点力 ----------
    Fx = [0.0] * (NUMNP + 1)                 # 1-based 索引
    for (a, b, c) in elems:
        x1, y1 = nodes[a - 1][:2]
        x2, y2 = nodes[b - 1][:2]
        x3, y3 = nodes[c - 1][:2]
        area   = 0.5 * abs((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
        F_elem = qx * area * t               # 元素总力
        for n in (a, b, c):
            Fx[n] += F_elem / 3.0            # 均分到 3 节点

    load_nodes = [(i, fx) for i, fx in enumerate(Fx) if i and abs(fx) > 1e-12]

    # ---------- 写文件 ----------
    lines = [f"Uniform_Plate_T3_{Nx}x{Ny}_Fx",
             f"{NUMNP}   1   {NLCASE}   {MODEX}"]

    for k, (x, y, bx, by, bz) in enumerate(nodes, 1):
        lines.append(f"{k}   {bx}   {by}   {bz}   {x:.6f}   {y:.6f}   0.0")

    lines.append(f"1   {len(load_nodes)}")
    for n, fx in load_nodes:
        lines.append(f"{n}   1   {fx:.6f}")   # DOF 1 → +x 方向

    lines.append(f"3   {NUMEG}   1")
    lines.append(f"1   {E:.1f}   {nu}   {t}")

    for eid, (a, b, c) in enumerate(elems, 1):
        lines.append(f"{eid}   {a}   {b}   {c}   1")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    # 生成 1×1 网格（两三角），文件名可自行修改
    Nx, Ny = 64,64
    filename = f"Uniform_Plate_T3_{Nx}x{Ny}.dat"
    write_uniform_plate_T3(filename, Nx, Ny)
