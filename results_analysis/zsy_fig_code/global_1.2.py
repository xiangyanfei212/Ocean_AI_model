import netCDF4 as nc
from fig.visualization import local_stream, global_stream
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import matplotlib as mpl
import scipy.io as sio
import warnings
from cartopy.util import add_cyclic_point
warnings.filterwarnings('ignore')

def create_map(longitude, latitude, variable, variable_u, variable_v, box, vmin=0, vmax= 10, cmap='Blues',
               q_step=8, xstep=5, ystep=3, lat_cff=10, lon_cff=10, title='T'):

    lon = longitude
    variable, longitude = add_cyclic_point(variable, coord=lon)
    variable_u, longitude = add_cyclic_point(variable_u, coord=lon)
    variable_v, longitude = add_cyclic_point(variable_v, coord=lon)
    longitude = longitude - 180

    X, Y = np.meshgrid(longitude, latitude)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    label_fontsize = 30
    title_fontsize = 30
    bar_label_fontsize = 30
    bar_tick_fontsize = 25
    scale = '50m'

    fig = plt.figure(figsize=(20, 12))
    proj = ccrs.PlateCarree(central_longitude=180)
    # proj = ccrs.Robinson(central_longitude=0)
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent(box, crs=proj)
    land = cfeature.NaturalEarthFeature('physical', 'land', scale, edgecolor='white', facecolor='white', zorder=2)
    ax.add_feature(land)
    ax.coastlines(scale, linewidth=1)
    ax.set_yticks(np.linspace(box[2] + lat_cff, box[3] - lat_cff, ystep), crs=proj)
    ax.set_xticks(np.linspace(box[0] + lon_cff, box[1] - lon_cff, xstep), crs=proj)
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=False))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_title(title, fontsize=title_fontsize, loc='center')
    ax.tick_params(which='major', direction='out', length=8, width=2, pad=0.3, labelsize=label_fontsize,
                   bottom=True, left=True, right=False, top=False)
    plot = ax.contourf(longitude, latitude, variable, cmap=cmap, norm=norm, transform=proj)
    q = ax.quiver(longitude[::q_step], latitude[::q_step], variable_u[::q_step, ::q_step], variable_v[::q_step, ::q_step],
                  width=0.005, headwidth=2, transform=proj)
    # ax.quiverkey(q, X=0.1, Y=0.98, U=1,
    #              label=r'1 m/s', labelpos='S', fontproperties={'size': 20})
    cbar = fig.colorbar(plot, ax=ax, shrink=0.7, pad=0.07, aspect=20, location='bottom', cmap=cmap, norm=norm)
    cbar.ax.set_xlabel('m/s', rotation=0, labelpad=10, fontsize=bar_label_fontsize)
    cbar.ax.tick_params(labelsize=bar_tick_fontsize)
    plt.show()

if __name__ == '__main__':
    f = nc.Dataset('/home/mae/mae_kg/fig/data_source/atom_variable.nc')
    print(f.variables.keys())
    longitude = f['longitude'][:]
    latitude = f['latitude'][:]
    u10 = f['u10'][0]
    v10 = f['v10'][0]
    msl = f['msl'][0]
    sst = f['sst'][0]
    uv_10 = np.sqrt(u10 ** 2 + v10 ** 2)
    box = [-180, 180, -90, 90]
    create_map(longitude, latitude, uv_10, u10, v10, box, vmin=0, vmax= 32, cmap='Blues',
               q_step=30, xstep=5, ystep=3, lat_cff=20, lon_cff=20, title='Surface wind speed')

