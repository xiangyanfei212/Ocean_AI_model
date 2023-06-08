import xarray as xr
import matplotlib.pyplot as plt

# 读取SST文件
ds = xr.open_dataset('sst.nc')

# 选择Kuroshio区域的数据
kuroshio_sst = ds.sel(lon=slice(120, 150), lat=slice(20, 40))

# 绘制海表温度图像
fig, ax = plt.subplots()
kuroshio_sst.sst.plot(ax=ax)

# 添加标题和标签
ax.set_title('Sea Surface Temperature in Kuroshio Region')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# 显示图像
plt.show()
