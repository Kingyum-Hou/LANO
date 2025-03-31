import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# 假设数据大小是 [90, 180]
data = np.random.rand(90, 180)  # 示例数据，实际应使用你的气象数据

# 创建一个图形
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 创建经纬度网格
lon = np.linspace(-180, 180, 180)
lat = np.linspace(-90, 90, 90)
lon_grid, lat_grid = np.meshgrid(lon, lat)

# 将经纬度转换为球面坐标
x = np.cos(np.radians(lat_grid)) * np.cos(np.radians(lon_grid))
y = np.cos(np.radians(lat_grid)) * np.sin(np.radians(lon_grid))
z = np.sin(np.radians(lat_grid))

# 使用数据值给每个点着色，使用渐变色
surf = ax.plot_surface(x, y, z, facecolors=cm.viridis(data), rstride=1, cstride=1, alpha=0.8, antialiased=True)

# 设置视角
ax.view_init(elev=15, azim=25)

# 添加颜色条
fig.colorbar(surf, shrink=0.5, aspect=5)

# 隐藏坐标轴
ax.set_axis_off()

# 正球体
ax.set_box_aspect([1, 1, 1])

# 显示图形
plt.savefig('1.png')
plt.clf()

