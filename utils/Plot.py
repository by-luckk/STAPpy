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
    Disp = np.array(disp) * scale

    if Disp.shape[1] > 3:
        # 提取平动位移（前3个分量）
        Disp_trans = Disp[:, :3]
        
    else:
        Disp_trans = Disp

    
    # 移动后的位置
    MovedCoords = Coords + Disp_trans

    # 绘图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 原始位置（蓝色）
    ax.scatter(Coords[:, 0], Coords[:, 1], Coords[:, 2], c='blue', label='Original Position', s=10)

    # 标记节点编号
    for i, (x, y, z) in enumerate(Coords):
        ax.text(x, y, z, f'{i + 1}', color='black', fontsize=8)

    # 移动后位置（红色）
    ax.scatter(MovedCoords[:, 0], MovedCoords[:, 1], MovedCoords[:, 2], c='red', label='Deformed Position', s=10)

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
            for i in range(NUME):
                element = EleGrp[i]
                nodes = element._nodes
                x = [node.XYZ[0] for node in nodes]
                y = [node.XYZ[1] for node in nodes]
                z = [node.XYZ[2] for node in nodes]
                nn = [node.NodeNumber - 1 for node in nodes]
                
                # 获取单元厚度（如果存在）
                thickness = 0.1
            
                # 原始位置
                x_closed = x + [x[0]]
                y_closed = y + [y[0]]
                z_closed = z + [z[0]]
                ax.plot(x_closed, y_closed, z_closed, color='blue', linewidth=1, alpha=0.5)
                
                # 变形后位置
                x_disp = x + Disp[nn, 0]
                y_disp = y + Disp[nn, 1]
                z_disp = z + Disp[nn, 2]
                x_disp_closed = np.append(x_disp, x_disp[0])
                y_disp_closed = np.append(y_disp, y_disp[0])
                z_disp_closed = np.append(z_disp, z_disp[0])
                ax.plot(x_disp_closed, y_disp_closed, z_disp_closed, 
                       color='red', linewidth=1.5)
                
                # 绘制厚度
               
                    # 计算上下表面坐标
                if element_type in ['MINDLIN_PLATE', 'MINDLIN_SHELL']:
                        # 对于板单元，考虑转动影响
                        theta_x = Disp[nn, 3] if Disp.shape[1] > 3 else 0
                        theta_y = Disp[nn, 4] if Disp.shape[1] > 4 else 0
                        
                        # 计算法向量
                        nx = -np.sin(theta_y) * np.cos(theta_x)
                        ny = np.sin(theta_x)
                        nz = np.cos(theta_y) * np.cos(theta_x)
                        
                        # 计算上下表面位置
                        x_top = x_disp + 0.5 * thickness * nx
                        y_top = y_disp + 0.5 * thickness * ny
                        z_top = z_disp + 0.5 * thickness * nz
                        
                        x_bottom = x_disp - 0.5 * thickness * nx
                        y_bottom = y_disp - 0.5 * thickness * ny
                        z_bottom = z_disp - 0.5 * thickness * nz
                else:
                        # 对于普通壳单元，只沿z方向加厚度
                        x_top = x_disp
                        y_top = y_disp
                        z_top = z_disp + 0.5 * thickness
                        
                        x_bottom = x_disp
                        y_bottom = y_disp
                        z_bottom = z_disp - 0.5 * thickness
                    
                    # 绘制上下表面
                for j in range(len(x_disp)):
                        ax.plot([x_bottom[j], x_top[j]], 
                               [y_bottom[j], y_top[j]], 
                               [z_bottom[j], z_top[j]], 
                               color='green', linewidth=0.5, alpha=0.7)
                    
                    # 绘制厚度侧面
                if len(x_disp) == 4:  # 四边形
                        sides = [(0,1), (1,2), (2,3), (3,0)]
                else:  # 三角形
                        sides = [(0,1), (1,2), (2,0)]
                    
                for side in sides:
                        # 底部边
                        ax.plot([x_bottom[side[0]], x_bottom[side[1]]], 
                               [y_bottom[side[0]], y_bottom[side[1]]], 
                               [z_bottom[side[0]], z_bottom[side[1]]], 
                               color='green', linewidth=0.5, alpha=0.7)
                        # 顶部边
                        ax.plot([x_top[side[0]], x_top[side[1]]], 
                               [y_top[side[0]], y_top[side[1]]], 
                               [z_top[side[0]], z_top[side[1]]], 
                               color='green', linewidth=0.5, alpha=0.7)
                        # 侧边
                        ax.plot([x_bottom[side[0]], x_top[side[0]]], 
                               [y_bottom[side[0]], y_top[side[0]]], 
                               [z_bottom[side[0]], z_top[side[0]]], 
                               color='green', linewidth=0.5, alpha=0.7)

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