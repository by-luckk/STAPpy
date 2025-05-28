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