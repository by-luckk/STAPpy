# 导入所需的库
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


def parse_stress_from_file_revised(filename):
    """
    一个健壮的文件解析函数，用于从给定的结果文件中提取SIGMA_XX应力值。

    Args:
        filename (str): 输入文件的路径。

    Returns:
        list: 一个包含所有单元SIGMA_XX值的浮点数列表。
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    stress_values = []
    in_stress_section = False
    for line in lines:
        if 'SIGMA_XX' in line and 'SIGMA_YY' in line:
            in_stress_section = True
            continue
        if in_stress_section and (line.strip() == '' or 'S O L U T I O N' in line):
            break
        if in_stress_section:
            if 'Stresses at element center' in line:
                continue
            parts = line.split()
            if len(parts) >= 2 and re.match(r'^[+\-]?\d+\.\d+e[+\-]?\d+$', parts[1]):
                stress_values.append(float(parts[1]))
    return stress_values


def run_analysis_with_theoretical_solution():
    """
    执行收敛性分析，与理论解进行比较，并绘制结果。
    """
    # =================================================================== #
    # 1. 定义理论解
    # =================================================================== #
    theoretical_solution = 1000.0

    files = {
        '1x1x1': 'H8\H8_Convergence_1x1x1.txt',
        '2x2x2': 'H8\H8_Convergence_2x2x2.txt',
        '4x4x4': 'H8\H8_Convergence_4x4x4.txt'
    }
    h_values = {'1x1x1': 1.0, '2x2x2': 0.5, '4x4x4': 0.25}

    avg_stresses = {}
    for name, filepath in files.items():
        stresses = parse_stress_from_file_revised(filepath)
        if stresses:
            avg_stresses[name] = np.mean(stresses)
        else:
            print(f"警告: 无法从文件 {filepath} 中提取应力数据。")
            avg_stresses[name] = 0

    # =================================================================== #
    # 2. 将所有算例结果与理论解比较，计算误差
    # =================================================================== #
    errors = {}
    h_plot_list = []
    error_plot_list = []

    # 遍历所有算例
    for name in files.keys():
        if name in avg_stresses:
            # 误差 = |计算值 - 理论解|
            error = abs(avg_stresses[name] - theoretical_solution)
            errors[name] = error
            # 只有当误差不为零时，才记录下来用于对数图和收敛阶计算
            if error > 1e-9:  # 使用一个很小的阈值来避免浮点数精度问题
                h_plot_list.append(h_values[name])
                error_plot_list.append(error)

    h_plot = np.array(h_plot_list)
    error_plot = np.array(error_plot_list)

    if len(h_plot) < 2:
        print("没有足够的数据点来进行收敛阶分析 (至少需要2个误差不为零的点)。")
        # 即使无法计算收敛阶，仍然可以绘制单个数据点
        if len(h_plot) == 1:
            plt.figure(figsize=(10, 6))
            plt.loglog(h_plot, error_plot, 'o', markersize=10, label='Calculated Error')
            plt.title('Convergence Analysis vs. Theoretical Solution', fontsize=16)
            plt.xlabel('Element Size (h) [log scale]', fontsize=12)
            plt.ylabel('Absolute Error vs. Theoretical Solution [log scale]', fontsize=12)
            plt.gca().invert_xaxis()
            plt.legend()
            plt.grid(True, which="both", ls="--")
            plt.show()
    else:
        # 基于所有可用的点计算收敛阶
        log_h = np.log(h_plot)
        log_error = np.log(error_plot)
        slope, intercept, r_value, p_value, std_err = linregress(log_h, log_error)
        convergence_order = slope

        # --- 结果可视化 ---
        plt.figure(figsize=(10, 6))
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.loglog(h_plot, error_plot, 'o', markersize=10, markerfacecolor='C0', markeredgecolor='k',
                   label='Calculated Error')
        fit_line = np.exp(intercept) * (h_plot ** convergence_order)
        plt.loglog(h_plot, fit_line, 'r--', label=f'Fitted Line (Slope = {convergence_order:.2f})')
        plt.title('Convergence Analysis vs. Theoretical Solution', fontsize=16)
        plt.xlabel('Element Size (h) [log scale]', fontsize=12)
        plt.ylabel('Absolute Error vs. Theoretical Solution [log scale]', fontsize=12)
        plt.gca().invert_xaxis()
        plt.legend(fontsize=12)
        plt.grid(True, which="both", ls="--")
        plt.show()

        print("\n" + "=" * 40)
        print(f"计算出的收敛阶 (拟合直线斜率): {convergence_order:.4f}")
        print(f"线性回归的R平方值 (拟合优度): {r_value ** 2:.4f}")
        print("=" * 40)

    # --- 在控制台打印所有算例的误差信息 ---
    print("\n--- 与理论解的误差分析结果 ---")
    print(f"理论解: {theoretical_solution:.4f}")
    for name in sorted(errors.keys()):
        print(f"网格: {name}, h: {h_values[name]}, 平均应力: {avg_stresses.get(name, 0):.4f}, 误差: {errors[name]:.4e}")


if __name__ == '__main__':
    run_analysis_with_theoretical_solution()