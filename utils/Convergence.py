import numpy as np

# ----------------------------------------------------------------------
# 1) 形函数 + 高斯点
# ----------------------------------------------------------------------
def shape_T3(xi, eta):
    return np.array([1.0 - xi - eta, xi, eta])

GAUSS_TRI_3 = np.array([
    [1/6, 1/6, 1/6],
    [2/3, 1/6, 1/6],
    [1/6, 2/3, 1/6],
])

# ----------------------------------------------------------------------
# 2) 理论解析位移  u(x,y), v(x,y)
# ----------------------------------------------------------------------
def exact_disp(x, y, p):
    """
    Saint-Venant 解：均匀体力 fx = qx (N/m²) 作用，
    左端固支，其余自由；平面应力。
    """
    L, qx, E, nu = p["L"], p["qx"], p["E"], p["nu"]
    u = qx / E * (L * x - x**2 / 2)  # 积分得到 u
    v = -nu * u                      # 泊松效应
    return np.array([u, v])

# ----------------------------------------------------------------------
# 3) L² 误差计算
# ----------------------------------------------------------------------
def l2_error(nodes, elems, disp, params):
    err_sq = 0.0
    for a, b, c in elems:
        xy  = nodes[[a, b, c]]               # (3,2)
        uh  = disp[[a, b, c], 0]             # u_h at 3 nodes
        vh  = disp[[a, b, c], 1]             # v_h at 3 nodes

        # Jacobian 常数 = 2*Area
        x1, y1 = xy[0]; x2, y2 = xy[1]; x3, y3 = xy[2]
        detJ = abs((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))

        for xi, eta, w in GAUSS_TRI_3:
            N   = shape_T3(xi, eta)
            x, y = N @ xy                     # 物理坐标
            uh_gp = N @ uh
            vh_gp = N @ vh
            ue, ve = exact_disp(x, y, params)
            err_sq += w * detJ * ((uh_gp - ue)**2 + (vh_gp - ve)**2)

    return np.sqrt(err_sq)

# ----------------------------------------------------------------------
# 4) 示例调用（把你的 FEM 结果替换掉 nodes/elems/disp 即可）
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # --- 示例：最粗 1×1 网格，两三角 ----
    nodes = np.array([[0,0],
                      [0,1],
                      [1,0],
                      [1,1]], float)
    elems = np.array([[0,1,2],
                      [1,3,2]], int)          # 0-based
    disp  = np.zeros((4,3))                  # 先用零位移占位

    params = dict(L=1.0, H=1.0, qx=1000.0,
                  E=2100.0, nu=0.3, t=1.0)

    L2 = l2_error(nodes, elems, disp, params)
    print(f"L² displacement error = {L2:.6e} m")
