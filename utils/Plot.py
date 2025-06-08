import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from element.ElementGroup import ElementTypes

def set_axes_equal(ax):
    '''Make 3D plot axes have equal scale'''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def PlotDisp(Coords, disp, scale=1.0, out_dir="output"):
    os.makedirs(out_dir, exist_ok=True)

    # 位移量（确保为 numpy 数组）
    disp_fixed = [row[:3] + [0.0] * (3 - len(row[:3])) for row in disp]
    Disp = np.array(disp_fixed) * scale
    MovedCoords = Coords + Disp

    # 绘图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 标记节点编号
    for i, (x, y, z) in enumerate(Coords):
        ax.text(x, y, z, f'{i + 1}', color='black', fontsize=8)

    # # 位移向量（箭头）
    # ax.quiver(
    #     Coords[:, 0], Coords[:, 1], Coords[:, 2],  # 起点
    #     Disp[:, 0], Disp[:, 1], Disp[:, 2],       # 向量分量
    #     color='green', normalize=False, linewidth=0.5, label='Displacement Vector'
    # )

    # plot mesh
    from Domain import Domain
    FEMData = Domain()
    # get mesh
    NUMEG = FEMData.GetNUMEG()
    for ELeGrpIndex in range(NUMEG):
        EleGrp = FEMData.GetEleGrpList()[ELeGrpIndex]
        NUME = EleGrp.GetNUME()
        ElementType = EleGrp.GetElementType()
        element_type = ElementTypes.get(ElementType)

        # TODO: check if it works for all element types
        # Ideally, if nodes are defined in order, it should work for all element types
        if element_type == 'Bar':
            for i in range(NUME):
                element = EleGrp[i]
                nodes = element._nodes
                x = [node.XYZ[0] for node in nodes]
                y = [node.XYZ[1] for node in nodes]
                z = [node.XYZ[2] for node in nodes]
                nn = [node.NodeNumber - 1 for node in nodes]
                ax.plot(x, y, z, color='blue', linewidth=1)
                ax.plot(x+Disp[nn, 0], y+Disp[nn, 1], z+Disp[nn, 2], color='red', linewidth=1)
        elif element_type == 'H8':  # 假设 'H8' 是 ElementTypes 中的键名
            # H8单元的边连接定义 (基于局部节点编号 0-7)
            # 例如，一个常见的顺序：
            # 底面: 0-1, 1-2, 2-3, 3-0
            # 顶面: 4-5, 5-6, 6-7, 7-4
            # 侧棱: 0-4, 1-5, 2-6, 3-7
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
                (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
                (0, 4), (1, 5), (2, 6), (3, 7)  # 侧棱
            ]
            for i in range(NUME):
                element = EleGrp[i]
                nodes_obj = element._nodes  # 获取节点对象列表

                # 提取所有节点的原始坐标和变形后坐标
                original_coords_element = np.array([node.XYZ for node in nodes_obj])
                node_indices_global = [node.NodeNumber - 1 for node in nodes_obj]  # 全局节点索引 (0-based)

                # 获取这些节点的位移
                disp_element_nodes = Disp[node_indices_global, :]
                deformed_coords_element = original_coords_element + disp_element_nodes

                for edge in edges:
                    # 原始网格
                    x_orig = [original_coords_element[edge[0]][0], original_coords_element[edge[1]][0]]
                    y_orig = [original_coords_element[edge[0]][1], original_coords_element[edge[1]][1]]
                    z_orig = [original_coords_element[edge[0]][2], original_coords_element[edge[1]][2]]
                    ax.plot(x_orig, y_orig, z_orig, color='blue', linewidth=0.8)

                    # 变形后网格
                    x_def = [deformed_coords_element[edge[0]][0], deformed_coords_element[edge[1]][0]]
                    y_def = [deformed_coords_element[edge[0]][1], deformed_coords_element[edge[1]][1]]
                    z_def = [deformed_coords_element[edge[0]][2], deformed_coords_element[edge[1]][2]]
                    ax.plot(x_def, y_def, z_def, color='red', linewidth=0.8)
        if element_type == 'Q4' or element_type == 'T3':
            for i in range(NUME):
                element = EleGrp[i]
                nodes = element._nodes
                x = [node.XYZ[0] for node in nodes]
                y = [node.XYZ[1] for node in nodes]
                z = [node.XYZ[2] for node in nodes]
                nn = [node.NodeNumber - 1 for node in nodes]
                
                x_closed = x + [x[0]]
                y_closed = y + [y[0]]
                z_closed = z + [z[0]]
                
                ax.plot(x_closed, y_closed, z_closed, color='blue', linewidth=1)
                
                x_disp = x + Disp[nn, 0]
                y_disp = y + Disp[nn, 1]
                z_disp = z + Disp[nn, 2]
                x_disp_closed = np.append(x_disp, x_disp[0])
                y_disp_closed = np.append(y_disp, y_disp[0])
                z_disp_closed = np.append(z_disp, z_disp[0])
                
                ax.plot(x_disp_closed, y_disp_closed, z_disp_closed, color='red', linewidth=1)

        if element_type in ['Beam', 'Frame']:
            for i in range(NUME):
                element = EleGrp[i]
                nodes = element._nodes
                nn = [node.NodeNumber - 1 for node in nodes]
                
                # 原始坐标
                x_orig = [node.XYZ[0] for node in nodes]
                y_orig = [node.XYZ[1] for node in nodes]
                z_orig = [node.XYZ[2] for node in nodes]
                
                                # 提取平动位移（前3个分量）
                if Disp.shape[1]> 3:
                    disp_trans = Disp[nn, :3]
                else:
                    disp_trans = Disp[nn]
                
                # 变形后坐标
                x_def = [x_orig[j] + disp_trans[j, 0] for j in range(len(x_orig))]
                y_def = [y_orig[j] + disp_trans[j, 1] for j in range(len(y_orig))]
                z_def = [z_orig[j] + disp_trans[j, 2] for j in range(len(z_orig))]
                
                # 绘制梁中心线
                ax.plot(x_orig, y_orig, z_orig, color='blue', linewidth=1.5)
                ax.plot(x_def, y_def, z_def, color='red', linewidth=1.5)
                


        if element_type == 'Plate':
            MovedCoords = []
            for i in range(NUME):
                element = EleGrp[i]
                nodes = element._nodes
                x = [node.XYZ[0] for node in nodes]
                y = [node.XYZ[1] for node in nodes]
                z = [node.XYZ[2] for node in nodes]
                nn = [node.NodeNumber - 1 for node in nodes]
        
                # 获取单元厚度（如果存在）
                thickness = getattr(element._ElementMaterial, 'thickness', 0.01)  # 默认0.1
        
                # 只考虑z方向的位移(w)和转角(θx, θy)
                w, theta_x, theta_y = [], [], []
                for n in nn:
                    if len(disp[n]) == 3 or len(disp[n]) == 5:
                        w.append(disp[n][-3])
                        theta_x.append(disp[n][-2])
                        theta_y.append(disp[n][-1])
                    else: # len(disp[n]) == 6
                        w.append(disp[n][2])
                        theta_x.append(disp[n][3])
                        theta_y.append(disp[n][4])
                w = np.array(w)
                theta_x = np.array(theta_x)
                theta_y = np.array(theta_y)

                # 变形后位置 - 只有z坐标变化
                x_disp = x + 0.5 * thickness * (-theta_y) 
                y_disp = y + 0.5 * thickness * theta_x
                z_disp = z + w  # 只有z方向有位移

                disp_single = np.column_stack((x_disp, y_disp, z_disp))
        
                # 原始位置（蓝色）
                x_closed = x + [x[0]]
                y_closed = y + [y[0]]
                z_closed = z + [z[0]]
                ax.plot(x_closed, y_closed, z_closed, color='blue', linewidth=1, alpha=0.5)
        
                # 变形后位置（红色）
                x_disp_closed = np.append(x_disp, x_disp[0])
                y_disp_closed = np.append(y_disp, y_disp[0])
                z_disp_closed = np.append(z_disp, z_disp[0])
                ax.plot(x_disp_closed, y_disp_closed, z_disp_closed, 
                color='red', linewidth=1.5)
        
                # 绘制厚度 - 考虑转角影响
                for j in range(len(x_disp)):
                    # 计算法向量变化（考虑转角）
                    # 假设小变形，转角θx和θy很小，可以使用近似
                    nx = -theta_y[j]  # 绕y轴转角影响x方向的法向量分量
                    ny = theta_x[j]   # 绕x轴转角影响y方向的法向量分量
                    nz = 1.0          # z方向为主分量
            
                    # 归一化法向量
                    norm = np.sqrt(nx**2 + ny**2 + nz**2)
                    nx /= norm
                    ny /= norm
                    nz /= norm
            
                    # 计算上下表面位置
                    x_top = x_disp[j] + 0.5 * thickness * nx
                    y_top = y_disp[j] + 0.5 * thickness * ny
                    z_top = z_disp[j] + 0.5 * thickness * nz
            
                    x_bottom = x_disp[j] - 0.5 * thickness * nx
                    y_bottom = y_disp[j] - 0.5 * thickness * ny
                    z_bottom = z_disp[j] - 0.5 * thickness * nz
            
                    # 绘制厚度线
                    ax.plot([x_bottom, x_top], 
                   [y_bottom, y_top], 
                   [z_bottom, z_top], 
                   color='green', linewidth=0.5, alpha=0.7)
        
                # 绘制厚度侧面（仅当有4个节点时）
                if len(x_disp) == 4:  # 四边形板单元
                 sides = [(0,1), (1,2), (2,3), (3,0)]
            
                for side in sides:
                    # 计算侧面上下点的法向量平均值
                    nx_avg = -(theta_y[side[0]] + theta_y[side[1]])/2
                    ny_avg = (theta_x[side[0]] + theta_x[side[1]])/2
                    nz_avg = 1.0
                    norm_avg = np.sqrt(nx_avg**2 + ny_avg**2 + nz_avg**2)
                    nx_avg /= norm_avg
                    ny_avg /= norm_avg
                    nz_avg /= norm_avg
                
                    # 底部边
                    x_bottom0 = x_disp[side[0]] - 0.5 * thickness * nx_avg
                    y_bottom0 = y_disp[side[0]] - 0.5 * thickness * ny_avg
                    z_bottom0 = z_disp[side[0]] - 0.5 * thickness * nz_avg
                
                    x_bottom1 = x_disp[side[1]] - 0.5 * thickness * nx_avg
                    y_bottom1 = y_disp[side[1]] - 0.5 * thickness * ny_avg
                    z_bottom1 = z_disp[side[1]] - 0.5 * thickness * nz_avg
                
                    # 顶部边
                    x_top0 = x_disp[side[0]] + 0.5 * thickness * nx_avg
                    y_top0 = y_disp[side[0]] + 0.5 * thickness * ny_avg
                    z_top0 = z_disp[side[0]] + 0.5 * thickness * nz_avg
                
                    x_top1 = x_disp[side[1]] + 0.5 * thickness * nx_avg
                    y_top1 = y_disp[side[1]] + 0.5 * thickness * ny_avg
                    z_top1 = z_disp[side[1]] + 0.5 * thickness * nz_avg
                
                    # 绘制侧面
                    ax.plot([x_bottom0, x_bottom1], 
                       [y_bottom0, y_bottom1], 
                       [z_bottom0, z_bottom1], 
                       color='green', linewidth=0.5, alpha=0.7)
                
                    ax.plot([x_top0, x_top1], 
                       [y_top0, y_top1], 
                       [z_top0, z_top1], 
                       color='green', linewidth=0.5, alpha=0.7)
                
                    ax.plot([x_bottom0, x_top0], 
                       [y_bottom0, y_top0], 
                       [z_bottom0, z_top0], 
                       color='green', linewidth=0.5, alpha=0.7)
                 
                MovedCoords.append(disp_single)
            MovedCoords = np.array(MovedCoords).reshape(-1, 3)

    # 绘制节点（在元素绘制完成后，确保节点可见）
    ax.scatter(Coords[:, 0], Coords[:, 1], Coords[:, 2], c='blue', label='Original Position', s=20, alpha=0.7)
    ax.scatter(MovedCoords[:, 0], MovedCoords[:, 1], MovedCoords[:, 2], c='red', label='Deformed Position', s=20)
    
    # 标记节点编号
    for i, (x, y, z) in enumerate(Coords):
        ax.text(x, y, z, f'{i + 1}', color='black', fontsize=8)


# 标签与图例
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"3D Nodal Displacement Vectors")
    ax.legend()
    ax.view_init(elev=30, azim=45)  # 设置视角，可调整
    set_axes_equal(ax)

    # 保存图片
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"displacement_vector.png"))
    plt.close(fig)