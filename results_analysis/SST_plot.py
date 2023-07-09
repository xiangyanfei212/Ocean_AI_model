import os
import h5py
import numpy as np
import netCDF4 as nc
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from var_channels import VAR_INDEX_15_LEVELS, LEVELS_15_STATS, LEVELS_15 

def plot_global_sst(
        pre_or_truth:str,        # 可视化预测还是真值，可选："pre"、"truth"
        projection:str,          # 投影方式，可选：'PlateCarree', 'robinson'
        file_path:str,           # 模型预报文件路径
        land_mask_file_path:str, # 海陆掩码文件路径
        interval:int,            # 经纬度间隔, 默认为1
        save_path:str,           # 图像保存路径
        init_condition:int,      # 第几次起报
        time_step:int,           # 第几个预报时间步
    ):

    var_idx = VAR_INDEX_15_LEVELS['T0']

    # read means and stds, for inverse transform data
    means = np.load(LEVELS_15_STATS['global_means_path'])
    stds  = np.load(LEVELS_15_STATS['global_stds_path'])
    means = means[0, var_idx, 0, 0]
    stds  = stds[0, var_idx, 0, 0]

    # read u, v
    tfid = nc.Dataset(file_path)
    if pre_or_truth == 'pre':
        data = tfid.variables['predicted'][init_condition, time_step, var_idx, :, :] 
    elif pre_or_truth == 'truth':
        data = tfid.variables['ground_truth'][init_condition, time_step, var_idx, :, :] 
    data = data * stds + means
    data = data[::interval,::interval]
    data = np.flipud(data)

    # set longitude and latitude
    longitude = np.linspace(-180,180,1441)[0:-1]  # 地图投影, 以180°为中心
    longitude = longitude[::interval]

    latitude = np.linspace(-90,90,721)[0:-1]
    latitude = latitude[::interval]
    longitude2d, latitude2d = np.meshgrid(longitude, latitude)

    longitude2d, latitude2d = np.meshgrid(longitude, latitude)

    if projection == 'PlateCarree':
        fig = plt.figure(figsize=(12,10))
        proj = ccrs.PlateCarree(central_longitude = 180) #central_longitude设置地图中心经度
        ax = fig.add_axes([0, 1, 0.6, 0.3],projection = proj) #设置图片位置和大小（左下宽高）
        ax.set_extent([0, 360.000001, -90, 90.000001], crs=ccrs.PlateCarree())#以下为经纬度设置
        
        xticks = np.arange(0, 360.1, 60)
        yticks = np.arange(-90, 90.1, 30)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, linestyle=':', color='k', alpha=0.8)###绘制经纬线###
        ax.set_xticks(xticks, crs=ccrs.PlateCarree()) ###绘制经纬度标记###
        ax.set_yticks(yticks, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.tick_params(labelsize=10)
        
        # land
        land = cfeature.NaturalEarthFeature('physical', 'land', scale='110m', edgecolor='black', facecolor='white',lw=0.5,zorder=2)#覆盖陆地
        ax.add_feature(land)
        
        # color
        mycolor=['#5D5199','#4E7DB3','#7BBAA9','#AFD9A9','#E0F2A4','#FCFEBF','#F9E69D','#F4BF79','#E78256','#CB544E','#A12643']
        cmap_color = mpl.colors.LinearSegmentedColormap.from_list('my_list', mycolor)
        cmap = cmap_color
        
        cf = ax.pcolormesh(longitude2d, latitude2d, data, cmap=cmap, vmin=-2.0, vmax=32.0, transform=ccrs.PlateCarree(central_longitude=180))
        
        # colorbar
        rect = [0.05,0.92,0.5,0.02]
        cbar_ax = fig.add_axes(rect)
        c=plt.colorbar(cf, cax=cbar_ax,extend="both",orientation='horizontal', aspect=20, pad=0.1)
        c.set_ticks(np.arange(-2, 32.00001,2)) #自定义数字间隔
        c.ax.tick_params(which='major',direction='in',labelsize=10,length=14) #色卡设置
        #c.set_label('SST',loc="center",fontsize=12,rotation=360)

        fig.savefig(save_path, dpi=2000, bbox_inches='tight', pad_inches=0.1)
    elif projection == 'robinson':
        fig = plt.figure(figsize=(12,10))

        # Use Robinson projection
        proj = ccrs.Robinson(central_longitude = 180)
        #proj = ccrs.PlateCarree(central_longitude = 180) #central_longitude设置地图中心经度
        
        ax = fig.add_axes([0, 1, 0.6, 0.3], projection = proj)#设置图片位置和大小（左下宽高）
        ax.set_global()  # Set the extent to cover the whole globe
        
        land = cfeature.NaturalEarthFeature('physical', 'land', scale='110m', edgecolor='black', facecolor='white',lw=0.5,zorder=2)#覆盖陆地
        ax.add_feature(land)
        
        mycolor=['#5D5199','#4E7DB3','#7BBAA9','#AFD9A9','#E0F2A4','#FCFEBF','#F9E69D','#F4BF79','#E78256','#CB544E','#A12643']
        cmap_color = mpl.colors.LinearSegmentedColormap.from_list('my_list', mycolor)
        cmap = cmap_color
        
        # The range of Colorbar
        level = np.linspace(-2, 32.00001, 18)
        
        #cf = ax.contourf(longitude2d, latitude2d, data, cmap=cmap, levels=level, extend='both', transform=ccrs.PlateCarree(central_longitude = 180))
        cf = ax.pcolormesh(longitude2d, latitude2d, data, cmap=cmap, vmin=-2.0, vmax=32.0, transform=ccrs.PlateCarree(central_longitude=180))
        
        rect = [0.05,0.92,0.5,0.02]
        cbar_ax = fig.add_axes(rect)    
        c=plt.colorbar(cf, cax=cbar_ax,extend="both",orientation='horizontal', aspect=20, pad=0.1)
        c.set_ticks(np.arange(-2, 32.00001,2)) #自定义数字间隔
        c.ax.tick_params(which='major',direction='in',labelsize=10,length=14) #色卡设置
        fig.savefig(save_path, dpi=2000, bbox_inches='tight', pad_inches=0.1)

if __name__ == '__main__':

    file_dir = '/home/bingxing2/home/scx6115/Ocean_AI_model/exp_15_levels/Masked_AE_Ocean/20230628-131150/2_steps_finetune'
    file_path = os.path.join(file_dir, 'autoregressive_predictions.h5') # 模型预报文件路径
    land_mask_file_path = LEVELS_15_STATS['land_mask_file'] # 海陆掩码文件路径
    projection = 'PlateCarree' # 投影方式，可选："robinson"，"PlateCarree"
    pre_or_truth = 'truth'    # 可视化预报还是真值
    init_condition = 1      # 第几次起报
    interval  = 1           # downscale, 经纬度间隔, 默认为1
    time_step = 5           # 第几个预报时间步
    save_path = os.path.join(file_dir, f'global_sst_{projection}_init_{init_condition}_timestep_{time_step}_{pre_or_truth}.png')  # 图像保存路径
    plot_global_sst(file_path           = file_path,
                    projection          = projection,
                    pre_or_truth        = pre_or_truth,
                    land_mask_file_path = land_mask_file_path,
                    init_condition      = init_condition,
                    interval            = interval,
                    time_step           = time_step,
                    save_path           = save_path)
