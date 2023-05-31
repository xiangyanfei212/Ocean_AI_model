import os
import sys
import datetime
import numpy as np
import pandas as pd
import configparser
import xarray as xr
from icecream import ic
from scipy.interpolate import griddata


def read_and_interp(target_file, in_file, out_file):
    xrdt = xr.open_dataset(target_file)
    ic(xrdt)
    try:
        tar_lat = xrdt['latitude'].values
        tar_lon = xrdt['longitude'].values
    except:
        tar_lat = xrdt['lat'].values
        tar_lon = xrdt['lon'].values

    xrdt = xr.open_dataset(in_file)
    ic(xrdt)
    org_val = xrdt['Band1'].values
    org_lat = xrdt['latitude'].values
    org_lon = xrdt['longitude'].values
    nonan_indices = np.where(np.isnan(org_val))
    ic(nonan_indices)

    # new_data = xr.DataArray(org_val, coords={"lat": org_lat, "lon": org_lon}, dims=["lat", "lon"])
    new_data = xr.Dataset({"Band1": (("lat", "lon"), org_val)},
                          coords={"lat": org_lat, "lon": org_lon},
                          # dims = ["lat", "lon"],
                        )
    ic(new_data)

    interp_val = new_data.interp(lat=tar_lat, lon=tar_lon, method='nearest', kwargs={"fill_value": "extrapolate"})
    ic(interp_val)

    nonan_indices = np.where(np.isnan(interp_val['Band1'].values))
    ic(nonan_indices)

    interp_val.to_netcdf(out_file)

if __name__ == '__main__':
    # target_file = '/work/home/acrzcyisbk/HYCOM_monthly/ssh/degree_1/GLBv0.08_53X_archMN.2003_12_ssh.nc'
    # in_file = '/work/home/acrzcyisbk/ETOPO/global_topo_1m.nc' 
    # out_file = '/work/home/acrzcyisbk/ETOPO/global_topo_1degree.nc' 
    # read_and_interp(target_file, in_file, out_file)

    target_file = '/work/home/acrzcyisbk/ERA5_monthly/2m_temperature/degree_025/2m_temperature_200306.nc'
    in_file = '/work/home/acrzcyisbk/ETOPO/global_topo_1m.nc'
    out_file = '/work/home/acrzcyisbk/ETOPO/global_topo_025degree.nc'
    read_and_interp(target_file, in_file, out_file)

