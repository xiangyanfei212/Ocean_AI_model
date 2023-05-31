import os
import h5py
import numpy as np
from icecream import ic


def get_means_stds(years, sample_dir):
    global_means = np.zeros((1,30,1,1))
    global_stds = np.zeros((1,30,1,1))
    for ii, year in enumerate(years):
        
        with h5py.File(os.path.join(sample_dir, f'{year}.h5'), 'r') as f:
    
            year_means = np.mean(f['fields'], keepdims=True, axis = (0,2,3))
            global_means += year_means 
            ic(year_means.shape)
    
            year_stds = np.var(f['fields'], keepdims=True, axis = (0,2,3))
            global_stds += year_stds
    
    global_means = global_means/len(years)
    global_stds = np.sqrt(global_stds/len(years))
    
    np.save(os.path.join(sample_dir, 'global_means.npy'), global_means)
    np.save(os.path.join(sample_dir, 'global_stds.npy'), global_stds)
    
    print("means: ", global_means.shape)
    print("stds: ", global_stds.shape)


def get_min_max_value(year_range, sample_dir):

    for i, year in enumerate(year_range):
        with h5py.File(os.path.join(sample_dir, f'{year}.h5'), 'r') as f:
            if i == 0:
                max_global = np.max(f['fields'], axis=(0,2,3), keepdims=False)
                min_global = np.min(f['fields'], axis=(0,2,3), keepdims=False)
            else:
                max_local = np.max(f['fields'], axis=(0,2,3), keepdims=False)
                min_local = np.min(f['fields'], axis=(0,2,3), keepdims=False)

                max_global[max_global < max_local] = max_local[max_global < max_local]
                min_global[min_global > min_local] = min_local[min_global > min_local]

    np.save(os.path.join(sample_dir, 'global_maxs.npy'), max_global)
    np.save(os.path.join(sample_dir, 'global_mins.npy'), min_global)

    ic(max_global.shape, min_global.shape)


if __name__ == '__main__':

    # years = [1994, 1997, 2000, 2003, 2006, 2009, 2012]
    # sample_dir = '../sample_02/train'
    # get_means_stds(years, sample_dir)
    # get_min_max_value(years, sample_dir)

    years = [2001, 2003, 2005, 2007, 2009]
    sample_dir = '../sample_03/train'
    get_means_stds(years, sample_dir)
    get_min_max_value(years, sample_dir)
