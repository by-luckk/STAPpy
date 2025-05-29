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

    # 移动后的位置
    MovedCoords = Coords + Disp

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