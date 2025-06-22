# figure 1. for paper
import xarray as xr
import torch
import numpy as np
from dataloaders.era5 import add_patch_missing
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import cartopy.crs as ccrs
from matplotlib.patches import Rectangle


def plt_era5(data0, mask0, img_path):
    fig, ax = plt.subplots(figsize=(13, 7))
    cax = ax.imshow(data0, cmap='viridis')
    # plant mask on the image
    for i in range(0, mask0.shape[0] - 2, 3):
        for j in range(0, mask0.shape[1] - 2, 3):
            if not mask0[i, j]:
                    rect = patches.Rectangle((j, i), 3, 3, linewidth=0, edgecolor='none', facecolor='black')
                    ax.add_patch(rect)
    plt.savefig(img_path)
    plt.clf()


def plt_era5_3D(data0, img_path):
    fig = plt.figure(figsize=(8, 6))
    #ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111, projection=ccrs.Orthographic(0, 0))

    # 绘制地图背景和海岸线
    #ax.set_facecolor('pink')  # 设置背景色
    ax.coastlines()

    # 创建经纬度网格
    lon = np.linspace(-180, 180, 180)
    lat = np.linspace(-90, 90, 90)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # 将经纬度转换为球面坐标
    x = np.cos(np.radians(lat_grid)) * np.cos(np.radians(lon_grid))
    y = np.cos(np.radians(lat_grid)) * np.sin(np.radians(lon_grid))
    z = np.sin(np.radians(lat_grid))

    # 使用数据值给每个点着色，使用渐变色
    #surf = ax.plot_surface(x, y, z, facecolors=cm.viridis(data0), rstride=1, cstride=1, alpha=0.8, antialiased=True)
    mesh = ax.pcolormesh(lon_grid, lat_grid, data0, cmap=cm.viridis, transform=ccrs.PlateCarree())

    # 添加缺失区域
    #patch = Rectangle((-50, 20), 3, 3, linewidth=1, edgecolor='black', facecolor='black', transform=ccrs.PlateCarree())
    #ax.add_patch(patch)
    """
    mask0_sub = mask0[::3, ::3]
    missing_mask = np.where(mask0_sub==False)
    for i in range(len(missing_mask[0])):
        lon_idx = missing_mask[1][i]*3
        lat_idx = missing_mask[0][i]*3
        patch = Rectangle((lon[lon_idx], lat[lat_idx]), 3, 3, linewidth=0, edgecolor='black', facecolor='black', transform=ccrs.PlateCarree())
        ax.add_patch(patch)
    """
    # 设置视角
    #ax.view_init(elev=15, azim=25)

    # 添加颜色条
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.colorbar(mesh, ax=ax, shrink=0.5, aspect=5)

    # 隐藏坐标轴
    ax.set_axis_off()

    # 正球体
    #ax.set_box_aspect([1, 1, 1])

    plt.savefig(img_path)
    plt.clf()


space_size = [720, 1440]
downsample = 8
data_dir = "/data/jingren/repository/dataset/WeatherBench/era5_temperature.grib"
H, W = space_size[0], space_size[1]
# load_data
ds = xr.open_dataset(data_dir)
data = torch.tensor(np.array(ds["t"]))
h = int((H / downsample))
w = int((W / downsample))
Tn = 7 * int(data.shape[0] / 7)
missing_rate = 0.
data = data[:, :720, :]
data = data[:, ::downsample, ::downsample]
data = data[::200, ...].permute(1, 2, 0).reshape(-1, 13)
data_missing, mask = add_patch_missing(data.unsqueeze(dim=0), missing_rate, (h, w), patch_size=9)
data_missing  = data_missing.reshape(1, h, w, 13)

data0 = data_missing[0, :, :, :]


for i in range(7):
    data0i = data0[:, :, i]
    plt_era5_3D(data0i, f"imgs/era5_{i}.png")
