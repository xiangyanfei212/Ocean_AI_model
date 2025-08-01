import os
import sys
import time
import glob
import h5py
import wandb
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import OrderedDict
from numpy.core.numeric import False_
from icecream import ic

import torch
import torchvision
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
from torchvision.utils import save_image
from torch.nn.parallel import DistributedDataParallel

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
from utils.YParams import YParams
from utils.data_loader_multifiles import get_downstream_data_loader
from utils.weighted_acc_rmse import weighted_rmse_torch_channels, weighted_acc_torch_channels, unweighted_acc_torch_channels, weighted_acc_masked_torch_channels

from utils import logging_utils
logging_utils.config_logger()

from networks.Masked_AE_Ocean import Masked_Ocean


def get_zonal_depth_mean(data):
    """
    data: [1, Lat, Lon]
    """
    zonal_mean = np.nanmean(data, axis=(0,3), keepdims=True)
    ic(zonal_mean.shape)
    zonal_mean = np.repeat(zonal_mean, data.shape[0], axis=0)
    zonal_mean = np.repeat(zonal_mean, data.shape[3], axis=3)

    depth_mean = np.nanmean(data, axis=(2,3), keepdims=True)
    ic(depth_mean.shape)
    depth_mean = np.repeat(depth_mean, data.shape[2], axis=2)
    depth_mean = np.repeat(depth_mean, data.shape[3], axis=3)

    assert len(np.where(np.isnan(depth_mean))[0]) == 0
    assert data.shape == zonal_mean.shape
    assert data.shape == depth_mean.shape

    return zonal_mean, depth_mean

def fill_land_with_0(data):

    with h5py.File(params.downstream_land_mask_path, 'r') as _f: 
        land_mask = torch.as_tensor(_f['fields'], dtype=bool)
        land_mask = np.array(land_mask).astype(int)

    land_mask = np.expand_dims(land_mask, (0,1))
    land_mask = np.repeat(land_mask, data.shape[1], axis=1)
    ic(data.shape, land_mask.shape)

    data = np.array(data.cpu())

    data[land_mask==0] = 0 

    return data

def fill_land_with_zonal_depth_mean(data):

    with h5py.File(params.downstream_land_mask_path, 'r') as _f: 
        land_mask = torch.as_tensor(_f['fields'], dtype=bool)
        land_mask = np.array(land_mask).astype(int)

    land_mask = np.expand_dims(land_mask, (0,1))
    land_mask = np.repeat(land_mask, data.shape[1], axis=1)
    ic(data.shape, land_mask.shape)

    data = np.array(data.cpu())

    data[land_mask==0] = np.nan
    zonal_mean, depth_mean = get_zonal_depth_mean(data)

    # fill land(nan) with zonal_mean
    land_indices = np.where(np.isnan(data))
    data[land_indices] = zonal_mean[land_indices]

    land_indices = np.where(np.isnan(data))
    data[land_indices] = depth_mean[land_indices]

    assert len(np.where(np.isnan(data))[0]) == 0

    return data

def gaussian_perturb(x, level=0.01, device=0):
    noise = level * torch.randn(x.shape).to(device, dtype=torch.float)
    return (x + noise)

def load_model(model, params, checkpoint_file):
    model.zero_grad()
    checkpoint_fname = checkpoint_file
    checkpoint = torch.load(checkpoint_fname)
    try:
        new_state_dict = OrderedDict()
        for key, val in checkpoint['model_state'].items():
            name = key[7:]
            if name != 'ged':
                new_state_dict[name] = val  
        model.load_state_dict(new_state_dict)
    except:
        model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model


def setup(params):
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    # get data loader
    valid_data_loader, valid_dataset = get_downstream_data_loader(
                params, 
                params.test_data_path, # backbone data
                params.downstream_test_data_path, # downstream data
                dist.is_initialized(), 
                train=False)

    # %% backbone data
    params.backbone_img_shape_x     = params.img_size_h
    params.backbone_img_shape_y     = params.img_size_w
    params.backbone_in_channels     = np.array(params.in_channels)
    params.backbone_out_channels    = np.array(params.out_channels)
    params.backbone_n_in_channels   = len(params.backbone_in_channels) 
    params.backbone_n_out_channels  = len(params.backbone_out_channels)
    if params["orography"]:
        params.backbone_n_in_channels += 1
    params.N_in_channels = params.backbone_n_in_channels # don't comment this
    params.N_out_channels = params.backbone_n_out_channels # don't comment this

    # %% downstream data
    params.downstream_img_shape_x    = valid_dataset.downstream_img_shape_x 
    params.downstream_img_shape_y    = valid_dataset.downstream_img_shape_y
    params.downstream_in_channels    = np.array(params.downstream_in_channels)
    params.downstream_out_channels   = np.array(params.downstream_out_channels)
    params.downstream_n_in_channels  = len(params.downstream_in_channels)
    params.downstream_n_out_channels = len(params.downstream_out_channels)

    # %% statistic data for normallization
    if params.normalization == 'zscore': 
        params.backbone_means = np.load(params.global_means_path)
        params.backbone_stds  = np.load(params.global_stds_path)
        params.downstream_means = np.load(params.downstream_global_means_path)
        params.downstream_stds  = np.load(params.downstream_global_stds_path)
    if params.normalization == 'minmax': 
        params.backbone_mins = np.load(params.global_mins_path)
        params.backbone_maxs = np.load(params.global_maxs_path)
        params.downstream_mins = np.load(params.downstream_global_mins_path)
        params.downstream_maxs = np.load(params.downstream_global_maxs_path)

    if params.nettype == 'BiochemicalNet':
        from downstream_networks.BiochemicalNet import BiochemicalNet as downstream_model
    else:
        raise Exception("not implemented")

    # init model
    backbone = Masked_Ocean(params)
    model = downstream_model(backbone, params)

    # Load model params
    logging.info('Loading trained model checkpoint from {}'.format(params['best_checkpoint_path']))
    model = load_model(model, params, params['best_checkpoint_path'])
    model = model.to(device)

    # load the validation data
    backbone_files_paths = glob.glob(params.test_data_path + "/*.h5")
    backbone_files_paths.sort()

    downstream_files_paths = glob.glob(params.downstream_test_data_path + "/*.h5")
    downstream_files_paths.sort()

    # which year
    yr = 0
    logging.info('Loading inference data')
    logging.info('Inference data from {}'.format(backbone_files_paths[yr]))
    backbone_valid_data_full = h5py.File(backbone_files_paths[yr], 'r')['fields']

    logging.info('Inference downstream data from {}'.format(downstream_files_paths[yr]))
    downstream_valid_data_full = h5py.File(downstream_files_paths[yr], 'r')['fields']

    return backbone_valid_data_full, downstream_valid_data_full, model

    
def autoregressive_inference(params, init_condition, backbone_valid_data_full, downstream_valid_data_full, model): 
    # initialize global variables
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    init_condition    = int(init_condition) 
    n_history         = params.n_history
    dt                = int(params.dt)
    prediction_length = int(params.prediction_length/dt)

    # initialize memory for image sequences and RMSE/ACC
    # %% backbone
    backbone_valid_loss             = torch.zeros((prediction_length, params.backbone_n_out_channels))
    backbone_acc                    = torch.zeros((prediction_length, params.backbone_n_out_channels))
    backbone_acc_unweighted         = torch.zeros((prediction_length, params.backbone_n_out_channels))
    backbone_valid_loss_coarse      = torch.zeros((prediction_length, params.backbone_n_out_channels))
    backbone_acc_coarse             = torch.zeros((prediction_length, params.backbone_n_out_channels))
    backbone_acc_coarse_unweighted  = torch.zeros((prediction_length, params.backbone_n_out_channels))
    backbone_seq_real = torch.zeros((prediction_length, params.backbone_n_out_channels, params.backbone_img_shape_x, params.backbone_img_shape_y))
    backbone_seq_pred = torch.zeros((prediction_length, params.backbone_n_out_channels, params.backbone_img_shape_x, params.backbone_img_shape_y))

    # %% downstream
    downstream_valid_loss             = torch.zeros((prediction_length, params.downstream_n_out_channels))
    downstream_acc                    = torch.zeros((prediction_length, params.downstream_n_out_channels))
    downstream_acc_unweighted         = torch.zeros((prediction_length, params.downstream_n_out_channels))
    downstream_valid_loss_coarse      = torch.zeros((prediction_length, params.downstream_n_out_channels))
    downstream_acc_coarse             = torch.zeros((prediction_length, params.downstream_n_out_channels))
    downstream_acc_coarse_unweighted  = torch.zeros((prediction_length, params.downstream_n_out_channels))
    downstream_seq_real = torch.zeros((prediction_length, params.downstream_n_out_channels, params.downstream_img_shape_x, params.downstream_img_shape_y))
    downstream_seq_pred = torch.zeros((prediction_length, params.downstream_n_out_channels, params.downstream_img_shape_x, params.downstream_img_shape_y))

    # extract valid data 
    backbone_valid_data = backbone_valid_data_full[init_condition:(init_condition+prediction_length*dt+n_history*dt):dt, :, 0:720]
    logging.info(f'backbone_valid_data_full: {backbone_valid_data_full.shape}')
    logging.info(f'backbone_valid_data: {backbone_valid_data.shape}')

    downstream_valid_data = downstream_valid_data_full[init_condition:(init_condition+prediction_length*dt+n_history*dt):dt]
    logging.info(f'downstream_valid_data_full: {downstream_valid_data_full.shape}')
    logging.info(f'downstream_valid_data: {downstream_valid_data.shape}')
    
    if params.normalization == 'zscore': 
        backbone_valid_data = (backbone_valid_data - params.backbone_means)/params.backbone_stds
        downstream_valid_data = (downstream_valid_data - params.downstream_means)/params.downstream_stds
    if params.normalization == 'minmax': 
        backbone_valid_data = (backbone_valid_data - params.backbone_mins) / (params.backbone_maxs - params.backbone_mins)
        downstream_valid_data = (downstream_valid_data - params.downstream_mins) / (params.downstream_maxs - params.downstream_mins)

    backbone_valid_data = torch.as_tensor(backbone_valid_data)
    downstream_valid_data = torch.as_tensor(downstream_valid_data)

    # orography
    if params.orography and params.normalization == 'zscore': 
        orography_path = params.orography_norm_zscore_path
    if params.orography and params.normalization == 'maxmin': 
        orography_path = params.orography_norm_maxmin_path
    if params.orography:
        orog = torch.as_tensor(np.expand_dims(np.expand_dims(h5py.File(orography_path, 'r')['orog'][0:720], axis = 0), axis = 0))
        logging.info("orography loaded; shape:{}".format(orog.shape))

    # autoregressive inference
    logging.info('Begin autoregressive inference')
    with torch.no_grad():
        for i in range(backbone_valid_data.shape[0]): 
            if i==0: # start of sequence, t0 --> t1
                backbone_first  = backbone_valid_data[0:n_history+1]
                backbone_future = backbone_valid_data[n_history+1]

                downstream_first  = downstream_valid_data[0:n_history+1, params.downstream_in_channels]
                downstream_future = downstream_valid_data[n_history+1,   params.downstream_out_channels]

                print(f'backbone_first: {backbone_first.shape}')
                print(f'backbone_future: {backbone_future.shape}')
                print(f'downstream_first: {downstream_first.shape}')
                print(f'downstream_future: {downstream_future.shape}')

                for h in range(n_history+1):
                    backbone_seq_real[h] = backbone_first[h*params.backbone_n_in_channels : (h+1)*params.backbone_n_in_channels, params.backbone_out_channels, :720] # extract history from 1st 
                    backbone_seq_pred[h] = backbone_seq_real[h]

                    downstream_seq_real[h] = downstream_first[h*params.downstream_n_in_channels : (h+1)*params.downstream_n_in_channels, params.downstream_out_channels] # extract history from 1st 
                    print(f'backbone_seq_real[h]: {backbone_seq_real[h].shape}')
                    print(f'downstream_seq_real[h]: {downstream_seq_real[h].shape}')
                    downstream_seq_pred[h] = downstream_seq_real[h]

                if params.perturb:
                    backbone_first = gaussian_perturb(backbone_first, level=params.n_level, device=device) # perturb the ic
                    downstream_first = gaussian_perturb(downstream_first, level=params.n_level, device=device) 

                backbone_first = backbone_first.to(device, dtype=torch.float)
                downstream_first = downstream_first.to(device, dtype=torch.float)

                if params.orography:
                    backbone_first = torch.cat((backbone_first, orog.to(device, dtype=torch.float)), axis=1)
                    print(f'backbone_first: {backbone_first.shape}, downstream_first: {downstream_first.shape}')
                    backbone_future_pred, downstream_future_pred = model(backbone_first, downstream_first)
                else:
                    backbone_future_pred, downstream_future_pred = model(backbone_first, downstream_first)

            else: # t1 --> t2 --> t3 ....
                if i < prediction_length-1:
                    backbone_future = backbone_valid_data[n_history+i+1]
                    downstream_future = downstream_valid_data[n_history+i+1] # !!! channels=5

                backbone_future_pred = backbone_future_pred.to(device, dtype=torch.float)
                downstream_future_pred = downstream_future_pred.to(device, dtype=torch.float)

                backbone_future_force = torch.unsqueeze(backbone_future[params.backbone_n_out_channels:], dim=0).to(device, dtype=torch.float)
                backbone_future_pred = torch.cat((backbone_future_pred, backbone_future_force), axis=1)

                if params.orography:

                    # backbone input: 
                    # 1. last time backbone output (t+1)
                    # 2. orog
                    # 3. atmos force (t+2)
                    # backbone output: (t+2)
                    orog  = orog.to(device, dtype=torch.float)
                    backbone_future_pred = torch.cat((backbone_future_pred, orog), axis=1)

                # fill the land with laititude mean
                downstream_future_pred = fill_land_with_zonal_depth_mean(downstream_future_pred)
                # downstream_future_pred = fill_land_with_0(downstream_future_pred)
                downstream_future_pred = torch.as_tensor(downstream_future_pred).to(device, dtype=torch.float)

                # downstream input
                # last time downstream model output (t+1)
                # downstream output: (t+2)
                inf_one_step_start = time.time()
                backbone_future_pred, downstream_future_pred = model(backbone_future_pred, downstream_future_pred)
                inf_one_step_time = time.time() - inf_one_step_start

                logging.info(f'inference one step time: {inf_one_step_time}')


            if i < prediction_length - 1: # not on the last step
                backbone_seq_pred[n_history+i+1] = backbone_future_pred.cpu()
                backbone_seq_real[n_history+i+1] = backbone_future[:params.backbone_n_out_channels]
                backbone_history_stack = backbone_seq_pred[i+1:i+2+n_history]

                downstream_seq_pred[n_history+i+1] = downstream_future_pred.cpu()
                downstream_seq_real[n_history+i+1] = downstream_future[params.downstream_out_channels]
                downstream_history_stack = downstream_seq_pred[i+1:i+2+n_history]

            backbone_future_pred = backbone_history_stack
            downstream_future_pred = downstream_history_stack

            backbone_pred = torch.unsqueeze(backbone_seq_pred[i], 0)
            backbone_tar  = torch.unsqueeze(backbone_seq_real[i], 0)
            print('backbone_pred:', backbone_pred.shape, 'backbone_tar:', backbone_tar.shape)

            downstream_pred = torch.unsqueeze(downstream_seq_pred[i], 0)
            downstream_tar  = torch.unsqueeze(downstream_seq_real[i], 0)
            print('downstream_pred:', downstream_pred.shape, 'downstream_tar:', downstream_tar.shape)

            if params.land_mask:
                # 0:land, 1:ocean
                with h5py.File(params.land_mask_path, 'r') as _f: 
                    backbone_mask_data = torch.as_tensor(_f['fields'][:720], dtype=bool)
                with h5py.File(params.downstream_land_mask_path, 'r') as _f: 
                    downstream_mask_data = torch.as_tensor(_f['fields'], dtype=bool)

                backbone_pred = torch.masked_fill(input=backbone_pred, mask=~backbone_mask_data, value=0)
                backbone_tar  = torch.masked_fill(input=backbone_tar,  mask=~backbone_mask_data, value=0)

                downstream_pred = torch.masked_fill(input=downstream_pred, mask=~downstream_mask_data, value=0)
                downstream_tar  = torch.masked_fill(input=downstream_tar,  mask=~downstream_mask_data, value=0)

            # Compute metrics 
            if params.normalization == 'zscore': 
                backbone_valid_loss[i] = weighted_rmse_torch_channels(backbone_pred, backbone_tar) * params.backbone_stds[:,params.backbone_out_channels,0,0]
                downstream_valid_loss[i] = weighted_rmse_torch_channels(downstream_pred, downstream_tar) * params.downstream_stds[:,params.downstream_out_channels,0,0]

            backbone_acc[i] = weighted_acc_torch_channels(backbone_pred, backbone_tar)
            downstream_acc[i] = weighted_acc_torch_channels(downstream_pred, downstream_tar)

            backbone_acc_unweighted[i] = unweighted_acc_torch_channels(backbone_pred, backbone_tar)
            downstream_acc_unweighted[i] = unweighted_acc_torch_channels(downstream_pred, downstream_tar)

            if params.interp > 0:   # TODO
                pred = downsample(pred, scale=params.interp)
                tar  = downsample(tar, scale=params.interp)

                if params.normalization == 'zscore': 
                    valid_loss_coarse[i] = weighted_rmse_torch_channels(pred, tar) * params.stds[:,:n_out_channels]

                acc_coarse[i] = weighted_acc_torch_channels(pred, tar)
                acc_coarse_unweighted[i] = unweighted_acc_torch_channels(pred, tar)

    backbone_seq_real               = backbone_seq_real.numpy()
    backbone_seq_pred               = backbone_seq_pred.numpy()
    backbone_valid_loss             = backbone_valid_loss.numpy()
    backbone_acc                    = backbone_acc.numpy()
    backbone_acc_unweighted         = backbone_acc_unweighted.numpy()
    backbone_acc_coarse             = backbone_acc_coarse.numpy()
    backbone_acc_coarse_unweighted  = backbone_acc_coarse_unweighted.numpy()
    backbone_valid_loss_coarse      = backbone_valid_loss_coarse.numpy()

    downstream_seq_real               = downstream_seq_real.numpy()
    downstream_seq_pred               = downstream_seq_pred.numpy()
    downstream_valid_loss             = downstream_valid_loss.numpy()
    downstream_acc                    = downstream_acc.numpy()
    downstream_acc_unweighted         = downstream_acc_unweighted.numpy()
    downstream_acc_coarse             = downstream_acc_coarse.numpy()
    downstream_acc_coarse_unweighted  = downstream_acc_coarse_unweighted.numpy()
    downstream_valid_loss_coarse      = downstream_valid_loss_coarse.numpy()

    return (np.expand_dims(backbone_seq_real[n_history:], 0), 
            np.expand_dims(backbone_seq_pred[n_history:], 0), 
            np.expand_dims(backbone_valid_loss,0), 
            np.expand_dims(backbone_acc, 0),
            np.expand_dims(backbone_acc_unweighted, 0), 
            np.expand_dims(backbone_valid_loss_coarse, 0), 
            np.expand_dims(backbone_acc_coarse, 0),
            np.expand_dims(backbone_acc_coarse_unweighted, 0),
            np.expand_dims(downstream_seq_real[n_history:], 0), 
            np.expand_dims(downstream_seq_pred[n_history:], 0), 
            np.expand_dims(downstream_valid_loss,0), 
            np.expand_dims(downstream_acc, 0),
            np.expand_dims(downstream_acc_unweighted, 0), 
            np.expand_dims(downstream_valid_loss_coarse, 0), 
            np.expand_dims(downstream_acc_coarse, 0),
            np.expand_dims(downstream_acc_coarse_unweighted, 0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", default='', type=str)
    parser.add_argument("--downstream_config", default='BiochemicalNet', type=str)
    parser.add_argument("--prediction_length", default=30, type=int)
    parser.add_argument("--decorrelation_time", default=30, type=int)
    parser.add_argument("--n_samples_per_year", default=365, type=int)
    parser.add_argument("--save_backbone_output", default=False, type=bool)
    args = parser.parse_args()

    config_path = os.path.join(args.exp_dir, 'config.yaml')
    params = YParams(config_path, args.downstream_config)

    params['downstream_config']    = args.downstream_config
    params['prediction_length']  = args.prediction_length
    params['decorrelation_time'] = args.decorrelation_time
    params['n_samples_per_year'] = args.n_samples_per_year
    params['resuming']           = False
    params['interp']             = 0 
    params['world_size']         = 1
    params['local_rank']         = 0
    params['best_checkpoint_path'] = os.path.join(args.exp_dir, 'training_checkpoints/best_ckpt.tar')

    # set up logging
    logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(args.exp_dir, 'inference_out.log'))
    logging_utils.log_versions()
    params.log()

    if params["ics_type"] == 'default':
        num_samples = args.n_samples_per_year - args.prediction_length
        stop        = num_samples
        init_conditions         = np.arange(0, stop, args.decorrelation_time)
        n_ics       = len(init_conditions)
    elif params["ics_type"] == "datetime": # TODO
        date_strings = params["date_strings"]
        init_conditions = []
        if params.perturb: 
            # like ensemble forecast, 
            # for perturbations use a single date and create n_ics perturbations
            n_ics = params["n_perturbations"]
            date = date_strings[0]
            date_obj = datetime.strptime(date,'%Y-%m-%d %H:%M:%S') 
            day_of_year = date_obj.timetuple().tm_yday - 1
            hour_of_day = date_obj.timetuple().tm_hour
            hours_since_jan_01_epoch = 24*day_of_year + hour_of_day
            for ii in range(n_ics):
                init_conditions.append(int(hours_since_jan_01_epoch/6))
        else:
            for date in date_strings:
                date_obj = datetime.strptime(date,'%Y-%m-%d %H:%M:%S') 
                day_of_year = date_obj.timetuple().tm_yday - 1
                hour_of_day = date_obj.timetuple().tm_hour
                hours_since_jan_01_epoch = 24*day_of_year + hour_of_day
                init_conditions.append(int(hours_since_jan_01_epoch/6))
        n_ics = len(init_conditions)
    logging.info("Inference for {} initial conditions".format(n_ics))

    try:
      autoregressive_inference_filetag = params["inference_file_tag"]
    except:
      autoregressive_inference_filetag = ""
    if params.interp > 0:
        autoregressive_inference_filetag = "_coarse"

    # get data and models
    backbone_valid_data_full, downstream_valid_data_full, model = setup(params)

    # initialize lists for image sequences and RMSE/ACC
    backbone_valid_loss = []
    backbone_valid_loss_coarse = []

    backbone_acc = []
    backbone_acc_unweighted = []

    backbone_acc_coarse = []
    backbone_acc_coarse_unweighted = []

    backbone_seq_pred = []
    backbone_seq_real = []

    downstream_valid_loss = []
    downstream_valid_loss_coarse = []

    downstream_acc = []
    downstream_acc_unweighted = []

    downstream_acc_coarse = []
    downstream_acc_coarse_unweighted = []

    downstream_seq_pred = []
    downstream_seq_real = []

    # run autoregressive inference for multiple initial conditions
    save_path = os.path.join(args.exp_dir, 'autoregressive_predictions' + autoregressive_inference_filetag+ '.h5')
    for i, inic in enumerate(init_conditions):
        logging.info("Initial condition {} of {}".format(i+1, n_ics))
        backbone_sr, backbone_sp, backbone_vl, backbone_a, backbone_au, backbone_vc, backbone_ac, backbone_acu, downstream_sr, downstream_sp, downstream_vl, downstream_a, downstream_au, downstream_vc, downstream_ac, downstream_acu  = autoregressive_inference(params, inic, backbone_valid_data_full, downstream_valid_data_full, model)

        print(f'saving to {save_path}')
        if i ==0:
            f = h5py.File(save_path, 'w')

            if args.save_backbone_output:
                f.create_dataset("backbone_ground_truth",    
                                 data=backbone_sr, 
                                 maxshape=[None, args.prediction_length, params.backbone_n_out_channels, 
                                           params.backbone_img_shape_x, params.backbone_img_shape_y], 
                                 dtype=np.float32)
                f.create_dataset("backbone_predicted",       
                                 data=backbone_sp, 
                                 maxshape=[None, args.prediction_length, params.backbone_n_out_channels, 
                                           params.backbone_img_shape_x, params.backbone_img_shape_y], 
                                 dtype=np.float32)
                f.create_dataset("backbone_rmse",            
                                 data=backbone_vl, 
                                 maxshape=[None, args.prediction_length, params.backbone_n_out_channels], 
                                 dtype=np.float32)
                f.create_dataset("backbone_rmse_coarse",     
                                 data=backbone_vc, 
                                 maxshape=[None, args.prediction_length, params.backbone_n_out_channels], 
                                 dtype=np.float32)
                f.create_dataset("backbone_acc",             
                                 data=backbone_a, 
                                 maxshape=[None, args.prediction_length, 
                                 params.backbone_n_out_channels], 
                                 dtype=np.float32)
                f.create_dataset("backbone_acc_unweighted",  
                                 data=backbone_au, 
                                 maxshape=[None, args.prediction_length, params.backbone_n_out_channels], 
                                 dtype=np.float32)
                f.create_dataset("backbone_acc_coarse",      
                                 data=backbone_ac, 
                                 maxshape=[None, args.prediction_length, params.backbone_n_out_channels], 
                                 dtype=np.float32)
                f.create_dataset("backbone_acc_coarse_unweighted", 
                                 data=backbone_acu, 
                                 maxshape=[None, args.prediction_length, params.backbone_n_out_channels], 
                                 dtype=np.float32)

            f.create_dataset("downstream_ground_truth",    
                             data=downstream_sr, 
                             maxshape=[None, args.prediction_length, params.downstream_n_out_channels, 
                                       params.downstream_img_shape_x, params.downstream_img_shape_y], 
                             dtype=np.float32)
            f.create_dataset("downstream_predicted",       
                             data=downstream_sp, 
                             maxshape=[None, args.prediction_length, params.downstream_n_out_channels, 
                                       params.downstream_img_shape_x, params.downstream_img_shape_y], 
                             dtype=np.float32)
            f.create_dataset("downstream_rmse",            
                             data=downstream_vl, 
                             maxshape=[None, args.prediction_length, params.downstream_n_out_channels], 
                             dtype =np.float32)
            f.create_dataset("downstream_rmse_coarse",     
                             data=downstream_vc, 
                             maxshape=[None, args.prediction_length, params.downstream_n_out_channels], 
                             dtype =np.float32)
            f.create_dataset("downstream_acc",             
                             data=downstream_a, 
                             maxshape=[None, args.prediction_length, params.downstream_n_out_channels], 
                             dtype =np.float32)
            f.create_dataset("downstream_acc_unweighted",  
                             data=downstream_au, 
                             maxshape=[None, args.prediction_length, params.downstream_n_out_channels], 
                             dtype =np.float32)
            f.create_dataset("downstream_acc_coarse",      
                             data=downstream_ac, 
                             maxshape=[None, args.prediction_length, params.downstream_n_out_channels], 
                             dtype =np.float32)
            f.create_dataset("downstream_acc_coarse_unweighted", 
                             data=downstream_acu, 
                             maxshape=[None, args.prediction_length, params.downstream_n_out_channels], 
                             dtype =np.float32)
            f.close()
        else:
            f = h5py.File(save_path, 'a')

            if args.save_backbone_output:
                f["backbone_ground_truth"].resize((f["backbone_ground_truth"].shape[0] + 1), axis = 0)
                f["backbone_ground_truth"][-1:] = backbone_sr 

                f["backbone_predicted"].resize((f["backbone_predicted"].shape[0] + 1), axis = 0)
                f["backbone_predicted"][-1:] = backbone_sp 

                f["backbone_rmse"].resize((f["backbone_rmse"].shape[0] + 1), axis = 0)
                f["backbone_rmse"][-1:] = backbone_vl

                f["backbone_rmse_coarse"].resize((f["backbone_rmse_coarse"].shape[0] + 1), axis = 0)
                f["backbone_rmse_coarse"][-1:] = backbone_vc

                f["backbone_acc"].resize((f["backbone_acc"].shape[0] + 1), axis = 0)
                f["backbone_acc"][-1:] = backbone_a

                f["backbone_acc_coarse"].resize((f["backbone_acc_coarse"].shape[0] + 1), axis = 0)
                f["backbone_acc_coarse"][-1:] = backbone_ac

                f["backbone_acc_unweighted"].resize((f["backbone_acc_unweighted"].shape[0] + 1), axis = 0)
                f["backbone_acc_unweighted"][-1:] = backbone_au

                f["backbone_acc_coarse_unweighted"].resize((f["backbone_acc_coarse_unweighted"].shape[0] + 1), axis = 0)
                f["backbone_acc_coarse_unweighted"][-1:] = backbone_acu

            f["downstream_ground_truth"].resize((f["downstream_ground_truth"].shape[0] + 1), axis = 0)
            f["downstream_ground_truth"][-1:] = downstream_sr 

            f["downstream_predicted"].resize((f["downstream_predicted"].shape[0] + 1), axis = 0)
            f["downstream_predicted"][-1:] = downstream_sp 

            f["downstream_rmse"].resize((f["downstream_rmse"].shape[0] + 1), axis = 0)
            f["downstream_rmse"][-1:] = downstream_vl

            f["downstream_rmse_coarse"].resize((f["downstream_rmse_coarse"].shape[0] + 1), axis = 0)
            f["downstream_rmse_coarse"][-1:] = downstream_vc

            f["downstream_acc"].resize((f["downstream_acc"].shape[0] + 1), axis = 0)
            f["downstream_acc"][-1:] = downstream_a

            f["downstream_acc_coarse"].resize((f["downstream_acc_coarse"].shape[0] + 1), axis = 0)
            f["downstream_acc_coarse"][-1:] = downstream_ac

            f["downstream_acc_unweighted"].resize((f["downstream_acc_unweighted"].shape[0] + 1), axis = 0)
            f["downstream_acc_unweighted"][-1:] = downstream_au

            f["downstream_acc_coarse_unweighted"].resize((f["downstream_acc_coarse_unweighted"].shape[0] + 1), axis = 0)
            f["downstream_acc_coarse_unweighted"][-1:] = downstream_acu
