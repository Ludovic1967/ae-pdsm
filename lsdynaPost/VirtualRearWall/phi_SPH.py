import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib

# 定义参数
h_p = 0.10   # 示例值，根据需要修改
b_D = 15/(7*np.pi*h_p**2)   # 示例值，根据需要修改

# 定义SPH分段函数
def f_D_SPH(r, b_D, h_p):
    if 0 <= r < h_p:
        return b_D * ((2/3) - ((r/h_p)**2) + (1/2)*((r/h_p)**3))
    elif h_p <= r < 2*h_p:
        return b_D * ((1/6)*((2 - (r/h_p))**3))
    else:
        return 0.0

# 对r的范围进行采样
r_values = np.linspace(0, 3*h_p, 300)
f_values = np.array([f_D_SPH(r, b_D, h_p) for r in r_values])

# 定义正态分布的固定参数
mu = 0.0  # 均值

# 设置待搜索的参数范围
n_eta_vals = np.linspace(2, 10, 100)        # n_eta在[0,10]内取100个值
sigma_vals = np.linspace(0.008, 0.3, 100)        # sigma在[1e-4,1]内取100个值，避免sigma=0

# 初始化存储误差的数组
errors = np.zeros((len(n_eta_vals), len(sigma_vals)))

# 网格搜索：遍历所有参数组合，计算均方误差
for i, n_eta in enumerate(n_eta_vals):
    for j, sigma in enumerate(sigma_vals):
        # 使用当前参数计算正态分布值
        y_pred = n_eta * norm.pdf(r_values, loc=mu, scale=sigma)
        # 计算平方误差的和
        errors[i, j] = np.sum((f_values - y_pred) ** 2)

# 找到最优的参数组合（误差最小）
min_index = np.unravel_index(np.argmin(errors), errors.shape)
best_n_eta = n_eta_vals[min_index[0]]
best_sigma = sigma_vals[min_index[1]]
print("Best n_eta:", best_n_eta)
print("Best sigma:", best_sigma)

# 用最优参数计算最佳拟合曲线
y_best = best_n_eta * norm.pdf(r_values, loc=mu, scale=best_sigma)

# 绘制SPH函数与最佳拟合正态分布曲线

front = {'family': 'SimHei', 'size': 12}
matplotlib.rc('font', **front)
plt.figure(figsize=(10, 6))
plt.plot(r_values, f_values, label=r'$f_{D,SPH}(r)$')
plt.plot(r_values, y_best, label=f'Normal Fit (n_eta={best_n_eta:.3f}, sigma={best_sigma:.3f})', color='red')
plt.axvline(x=h_p, color='gray', linestyle='--', label=r'$r = h_p$')
plt.axvline(x=2*h_p, color='gray', linestyle='--', label=r'$r = 2h_p$')
plt.title('SPH函数与最佳拟合正态分布')
plt.xlabel('$r$')
plt.ylabel('函数值')
plt.legend()
plt.grid(True)
plt.show()

# 可视化网格搜索的误差热力图
plt.figure(figsize=(8, 6))
# 创建网格数据（注意：x轴为sigma，y轴为n_eta）
Sigma, N_eta = np.meshgrid(sigma_vals, n_eta_vals)
plt.contourf(Sigma, N_eta, errors, levels=50, cmap='viridis')
plt.colorbar(label='平方误差和')
plt.scatter([best_sigma], [best_n_eta], color='red', marker='x', s=100, label='最佳参数')
plt.xlabel('sigma')
plt.ylabel('n_eta')
plt.title('网格搜索误差热力图')
plt.legend()
plt.show()
