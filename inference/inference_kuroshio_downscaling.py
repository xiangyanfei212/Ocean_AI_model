


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
from utils.data_loader_multifiles import get_finetune_data_loader
from utils.weighted_acc_rmse import unweighted_rmse_torch_channels, weighted_rmse_torch_channels, weighted_acc_torch_channels, unweighted_acc_torch_channels, weighted_acc_masked_torch_channels

from utils import logging_utils
logging_utils.config_logger()

from networks.Masked_AE_Ocean import Masked_Ocean


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
    valid_data_loader, valid_dataset = get_finetune_data_loader(
                params, 
                params.test_data_path, # backbone data
                params.finetune_test_data_path, # downstream data
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

    # %% finetune data
    params.finetune_in_channels    = np.array(params.finetune_in_channels)
    params.finetune_out_channels   = np.array(params.finetune_out_channels)
    params.finetune_n_in_channels  = len(params.finetune_in_channels)
    params.finetune_n_out_channels = len(params.finetune_out_channels)

    # %% statistic data for normallization
    if params.normalization == 'zscore': 
        params.backbone_means = np.load(params.global_means_path)
        params.backbone_stds  = np.load(params.global_stds_path)
        params.finetune_means = np.load(params.finetune_global_means_path)
        params.finetune_stds  = np.load(params.finetune_global_stds_path)
        params.finetune_means = params.finetune_means[:,params.finetune_in_channels]
        params.finetune_stds  = params.finetune_stds[:,params.finetune_in_channels]
    if params.normalization == 'minmax': 
        params.backbone_mins = np.load(params.global_mins_path)
        params.backbone_maxs = np.load(params.global_maxs_path)
        params.finetune_mins = np.load(params.finetune_global_mins_path)
        params.finetune_maxs = np.load(params.finetune_global_maxs_path)
        params.finetune_mins = params.finetune_mins[:,params.finetune_in_channels]
        params.finetune_maxs = params.finetune_maxs[:,params.finetune_in_channels]

    if params.nettype == 'DownScalingNet_fourier':
        from finetune_networks.DownScalingNet_fourier import DownScalingNet as finetune_model
    elif params.nettype == 'DownScalingNet':
        from finetune_networks.DownScalingNet import DownScalingNet as finetune_model
    elif params.nettype == 'DownScalingNet_SR':
        from finetune_networks.DownScalingNet_SR import DownScalingNet as finetune_model
    else:
        raise Exception("not implemented")

    # init model
    backbone = Masked_Ocean(params)
    total_model = finetune_model(backbone, params)

    # Load model params
    logging.info('Loading trained model checkpoint from {}'.format(params['best_checkpoint_path']))
    total_model = load_model(total_model, params, params['best_checkpoint_path'])
    # total_model = total_model.to(device)

    logging.info('Loading trained backbone checkpoint from {}'.format(params['backbone_best_checkpoint_path']))
    backbone = Masked_Ocean(params)
    backbone = load_model(backbone, params, params['backbone_best_checkpoint_path'])
    # backbone = backbone.to(device)

    # load the validation data
    backbone_files_paths = glob.glob(params.test_data_path + "/*.h5")
    backbone_files_paths.sort()

    finetune_files_paths = glob.glob(params.finetune_test_data_path + "/*.h5")
    finetune_files_paths.sort()

    # which year
    yr = 0
    logging.info('Loading inference data')
    logging.info('Inference data from {}'.format(backbone_files_paths[yr]))
    backbone_valid_data_full = h5py.File(backbone_files_paths[yr], 'r')['fields']

    logging.info('Inference finetune data from {}'.format(finetune_files_paths[yr]))
    finetune_valid_data_0p25_full = h5py.File(finetune_files_paths[yr], 'r')['fields_0p25']
    finetune_valid_data_0p08_full = h5py.File(finetune_files_paths[yr], 'r')['fields_0p08']

    return backbone_valid_data_full, finetune_valid_data_0p25_full, finetune_valid_data_0p08_full, backbone, total_model

    
def autoregressive_inference(params, init_condition, backbone_valid_data_full, finetune_valid_data_0p25_full, finetune_valid_data_0p08_full, backbone, total_model): 

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

    # %% finetune
    finetune_rmse_weighted          = torch.zeros((prediction_length, params.finetune_n_out_chans))
    finetune_rmse_unweighted        = torch.zeros((prediction_length, params.finetune_n_out_chans))
    finetune_acc_weighted           = torch.zeros((prediction_length, params.finetune_n_out_chans))
    finetune_acc_unweighted         = torch.zeros((prediction_length, params.finetune_n_out_chans))
    finetune_acc_coarse_weighted    = torch.zeros((prediction_length, params.finetune_n_out_chans))
    finetune_acc_coarse_unweighted  = torch.zeros((prediction_length, params.finetune_n_out_chans))
    # a part of the backbone model's output, feed to the finetune model, 0p25, before downscaling
    finetune_seq_inp                = torch.zeros((prediction_length, params.finetune_n_in_chans, 
                                                   params.finetune_in_size_h, params.finetune_in_size_w)) 
    # 0p08, groud truth
    finetune_seq_real               = torch.zeros((prediction_length, params.finetune_n_out_chans, 
                                                   params.finetune_target_size_h, params.finetune_target_size_w))
    # 0p08, the finetune model's outputs, prediction
    finetune_seq_pred               = torch.zeros((prediction_length, params.finetune_n_out_chans, 
                                                   params.finetune_target_size_h, params.finetune_target_size_w)) 

    # extract valid data 
    backbone_valid_data = backbone_valid_data_full[init_condition:(init_condition+prediction_length*dt+n_history*dt):dt, :, 0:720]
    logging.info(f'backbone_valid_data_full: {backbone_valid_data_full.shape}') 
    logging.info(f'backbone_valid_data: {backbone_valid_data.shape}')

    finetune_valid_data_0p08 = finetune_valid_data_0p08_full[init_condition:(init_condition+prediction_length*dt+n_history*dt):dt, params.finetune_in_channels]
    logging.info(f'finetune_valid_data_0p08_full: {finetune_valid_data_0p08_full.shape}')
    logging.info(f'finetune_valid_data_0p08: {finetune_valid_data_0p08.shape}')

    finetune_valid_data_0p25 = finetune_valid_data_0p25_full[init_condition:(init_condition+prediction_length*dt+n_history*dt):dt, params.finetune_in_channels]
    logging.info(f'finetune_valid_data_0p25_full: {finetune_valid_data_0p25_full.shape}')
    logging.info(f'finetune_valid_data_0p25: {finetune_valid_data_0p25.shape}')
    
    if params.normalization == 'zscore': 
        backbone_valid_data = (backbone_valid_data - params.backbone_means)/params.backbone_stds
        finetune_valid_data_0p25 = (finetune_valid_data_0p25 - params.finetune_means)/params.finetune_stds
        finetune_valid_data_0p08 = (finetune_valid_data_0p08 - params.finetune_means)/params.finetune_stds
    if params.normalization == 'minmax': 
        backbone_valid_data = (backbone_valid_data - params.backbone_mins) / (params.backbone_maxs - params.backbone_mins)
        finetune_valid_data_0p25 = (finetune_valid_data_0p25 - params.finetune_mins) / (params.finetune_maxs - params.finetune_mins)
        finetune_valid_data_0p08 = (finetune_valid_data_0p08 - params.finetune_mins) / (params.finetune_maxs - params.finetune_mins)

    backbone_valid_data = torch.as_tensor(backbone_valid_data)
    finetune_valid_data_0p25 = torch.as_tensor(finetune_valid_data_0p25)
    finetune_valid_data_0p08 = torch.as_tensor(finetune_valid_data_0p08)

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
                print(f'backbone_first: {backbone_first.shape}')
                print(f'backbone_future: {backbone_future.shape}')

                finetune_future_0p08 = finetune_valid_data_0p08[n_history+1]
                print(f'finetune_future_0p08: {finetune_future_0p08.shape}')

                for h in range(n_history+1):
                    backbone_seq_real[h] = backbone_first[h*params.backbone_n_in_channels : (h+1)*params.backbone_n_in_channels, params.backbone_out_channels, :720] # extract history from 1st 
                    backbone_seq_pred[h] = backbone_seq_real[h]

                    finetune_seq_real[h] = finetune_future_0p08
                    # [h*params.finetune_n_in_channels : (h+1)*params.finetune_n_in_channels, params.finetune_out_channels] # extract history from 1st 
                    finetune_seq_pred[h] = finetune_seq_real[h]
                    print(f'backbone_seq_real[h]: {backbone_seq_real[h].shape}')
                    print(f'finetune_seq_real[h]: {finetune_seq_real[h].shape}')

                if params.perturb:
                    backbone_first      = gaussian_perturb(backbone_first, level=params.n_level, device=device) # perturb the ic
                    # finetune_first_0p25 = gaussian_perturb(finetune_first_0p25, level=params.n_level, device=device) 

                backbone_first = backbone_first.to(device, dtype=torch.float)
                # finetune_first_0p25 = finetune_first_0p25.to(device, dtype=torch.float)

                if params.orography:
                    backbone_first = torch.cat((backbone_first, orog.to(device, dtype=torch.float)), axis=1)

                # step 1
                backbone = backbone.to(device)
                backbone_future_pred_0 = backbone(backbone_first)

                # lat_range_0p25: [20, 50], lat_start_idx: 160, lat_end_idx: 280
                # lon_range_0p25: [122.5, 180], lon_start_idx: 490, lon_end_idx: 720
                if params.finetune_n_in_channels == 2:
                    backbone_future_pred_0 = backbone_future_pred_0[:,[30,45],160:280,490:720]
                else:
                    backbone_future_pred_0 = backbone_future_pred_0[:,[0,15,30,45,60],160:280,490:720]
                backbone = backbone.to('cpu')
                # del backbone
                torch.cuda.empty_cache()

                # step 2
                print(f'backbone_first: {backbone_first.shape}')
                print(f'backbone_future_pred_0: {backbone_future_pred_0.shape}')
                total_model = total_model.to(device)
                backbone_future_pred, finetune_future_pred = total_model(backbone_first, backbone_future_pred_0)
                print('finetune_future_pred:', finetune_future_pred.shape)
                total_model = total_model.to('cpu')
                torch.cuda.empty_cache()


            else: # t1 --> t2 --> t3 ....
                if i < prediction_length-1:
                    backbone_future      = backbone_valid_data[n_history+i+1]
                    finetune_future_0p08 = finetune_valid_data_0p08[n_history+i+1] 

                backbone_future_pred = backbone_future_pred.to(device, dtype=torch.float)
                backbone_future_force = torch.unsqueeze(backbone_future[params.backbone_n_out_channels:], dim=0).to(device, dtype=torch.float)
                backbone_future_pred = torch.cat((backbone_future_pred, backbone_future_force), axis=1)

                if params.orography:
                    orog  = orog.to(device, dtype=torch.float)
                    backbone_future_pred = torch.cat((backbone_future_pred, orog), axis=1)

                finetune_inp = {}

                # predict
                # step 1
                backbone = backbone.to(device)
                backbone_future_pred_0 = backbone(backbone_future_pred)
                if params.finetune_n_in_channels == 2:
                    backbone_future_pred_0 = backbone_future_pred_0[:,[30,45],160:280,490:720]
                else:
                    backbone_future_pred_0 = backbone_future_pred_0[:,[0,15,30,45,60],160:280,490:720]
                backbone = backbone.to('cpu')
                torch.cuda.empty_cache()

                # step 2
                total_model = total_model.to(device)
                inf_one_step_start = time.time()
                backbone_future_pred, finetune_future_pred = total_model(backbone_future_pred, backbone_future_pred_0) 
                print('finetune_future_pred:', finetune_future_pred.shape)
                inf_one_step_time = time.time() - inf_one_step_start
                total_model = total_model.to('cpu')
                torch.cuda.empty_cache()

                logging.info(f'inference one step time: {inf_one_step_time}')

            if i < prediction_length - 1: # not on the last step
                backbone_seq_pred[n_history+i+1] = backbone_future_pred.cpu()
                backbone_seq_real[n_history+i+1] = backbone_future[params.backbone_out_channels]
                backbone_history_stack = backbone_seq_pred[i+1:i+2+n_history]

                finetune_seq_pred[n_history+i+1] = finetune_future_pred.cpu()
                finetune_seq_real[n_history+i+1] = finetune_future_0p08
                finetune_history_stack = finetune_seq_pred[i+1:i+2+n_history]

            backbone_future_pred = backbone_history_stack
            finetune_future_pred = finetune_history_stack

            backbone_pred = torch.unsqueeze(backbone_seq_pred[i], 0)
            backbone_tar  = torch.unsqueeze(backbone_seq_real[i], 0)
            print('backbone_pred:', backbone_pred.shape, 'backbone_tar:', backbone_tar.shape)

            finetune_pred = torch.unsqueeze(finetune_seq_pred[i], 0)
            finetune_tar  = torch.unsqueeze(finetune_seq_real[i], 0)
            print('finetune_pred:', finetune_pred.shape, 'finetune_tar:', finetune_tar.shape)

            if params.land_mask:
                # 0:land, 1:ocean
                with h5py.File(params.land_mask_path, 'r') as _f: 
                    backbone_mask_data = torch.as_tensor(_f['fields'][:720], dtype=bool)
                with h5py.File(params.finetune_land_mask_path, 'r') as _f: 
                    finetune_mask_data = torch.as_tensor(_f['fields'], dtype=bool)
                    
                if params.finetune_config == 'DownScalingNet':
                    finetune_mask_data = finetune_mask_data[:,params.finetune_out_channels,:,:]

                backbone_pred = torch.masked_fill(input=backbone_pred, mask=~backbone_mask_data, value=0)
                backbone_tar  = torch.masked_fill(input=backbone_tar,  mask=~backbone_mask_data, value=0)

                finetune_pred = torch.masked_fill(input=finetune_pred, mask=~finetune_mask_data, value=0)
                finetune_tar  = torch.masked_fill(input=finetune_tar,  mask=~finetune_mask_data, value=0)

            # Compute metrics 
            if params.normalization == 'zscore': 
                backbone_valid_loss[i]      = weighted_rmse_torch_channels(backbone_pred, backbone_tar) * params.backbone_stds[:,params.backbone_out_channels,0,0]
                print('params.finetune_stds:', params.finetune_stds.shape)
                print('weighted_rmse_torch_channels(finetune_pred, finetune_tar):', weighted_rmse_torch_channels(finetune_pred, finetune_tar).shape)
                finetune_rmse_weighted[i]   = weighted_rmse_torch_channels(finetune_pred, finetune_tar) * params.finetune_stds[:,:,0,0]
                finetune_rmse_unweighted[i] = unweighted_rmse_torch_channels(finetune_pred, finetune_tar) * params.finetune_stds[:,:,0,0]

            backbone_acc[i]          = weighted_acc_torch_channels(backbone_pred, backbone_tar)
            finetune_acc_weighted[i] = weighted_acc_torch_channels(finetune_pred, finetune_tar)

            backbone_acc_unweighted[i] = unweighted_acc_torch_channels(backbone_pred, backbone_tar)
            finetune_acc_unweighted[i] = unweighted_acc_torch_channels(finetune_pred, finetune_tar)

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

    finetune_seq_real               = finetune_seq_real.numpy()
    finetune_seq_pred               = finetune_seq_pred.numpy()
    finetune_rmse_weighted          = finetune_rmse_weighted.numpy()
    finetune_rmse_unweighted        = finetune_rmse_unweighted.numpy()
    finetune_acc_weighted           = finetune_acc_weighted.numpy()
    finetune_acc_unweighted         = finetune_acc_unweighted.numpy()
    finetune_acc_coarse_weighted    = finetune_acc_coarse_weighted.numpy()
    finetune_acc_coarse_unweighted  = finetune_acc_coarse_unweighted.numpy()

    return (np.expand_dims(backbone_seq_real[n_history:], 0), 
            np.expand_dims(backbone_seq_pred[n_history:], 0), 
            np.expand_dims(backbone_valid_loss,0), 
            np.expand_dims(backbone_acc, 0),
            np.expand_dims(backbone_acc_unweighted, 0), 
            np.expand_dims(backbone_valid_loss_coarse, 0), 
            np.expand_dims(backbone_acc_coarse, 0),
            np.expand_dims(backbone_acc_coarse_unweighted, 0),
            np.expand_dims(finetune_seq_real[n_history:], 0), 
            np.expand_dims(finetune_seq_pred[n_history:], 0), 
            np.expand_dims(finetune_rmse_weighted, 0), 
            np.expand_dims(finetune_rmse_unweighted, 0), 
            np.expand_dims(finetune_acc_weighted, 0),
            np.expand_dims(finetune_acc_unweighted, 0), 
            np.expand_dims(finetune_acc_coarse_weighted, 0),
            np.expand_dims(finetune_acc_coarse_unweighted, 0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", default='', type=str)
    parser.add_argument("--backbone_dir", default='', type=str)
    parser.add_argument("--finetune_config", default='DownScalingNet', type=str)
    parser.add_argument("--prediction_length", default=30, type=int)
    parser.add_argument("--decorrelation_time", default=30, type=int)
    parser.add_argument("--n_samples_per_year", default=365, type=int)
    parser.add_argument("--save_backbone_output", default=False, type=bool)
    args = parser.parse_args()

    config_path = os.path.join(args.exp_dir, 'config.yaml')
    print('config_path:', config_path)
    params = YParams(config_path, args.finetune_config)

    params['finetune_config']    = args.finetune_config
    params['prediction_length']  = args.prediction_length
    params['decorrelation_time'] = args.decorrelation_time
    params['n_samples_per_year'] = args.n_samples_per_year
    params['resuming']           = False
    params['interp']             = 0 
    params['world_size']         = 1
    params['local_rank']         = 0
    params['best_checkpoint_path']          = os.path.join(args.exp_dir, 'training_checkpoints/best_ckpt.tar')
    params['backbone_best_checkpoint_path'] = os.path.join(args.backbone_dir, 'training_checkpoints/best_ckpt.tar')

    # set up logging
    logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(args.exp_dir, 'inference_out.log'))
    logging_utils.log_versions()
    params.log()

    if params["ics_type"] == 'default':
        num_samples = args.n_samples_per_year - args.prediction_length
        stop        = num_samples
        ics         = np.arange(0, stop, args.decorrelation_time)
        n_ics       = len(ics)
    elif params["ics_type"] == "datetime":
        date_strings = params["date_strings"]
        ics = []
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
                ics.append(int(hours_since_jan_01_epoch/6))
        else:
            for date in date_strings:
                date_obj = datetime.strptime(date,'%Y-%m-%d %H:%M:%S') 
                day_of_year = date_obj.timetuple().tm_yday - 1
                hour_of_day = date_obj.timetuple().tm_hour
                hours_since_jan_01_epoch = 24*day_of_year + hour_of_day
                ics.append(int(hours_since_jan_01_epoch/6))
        n_ics = len(ics)
    logging.info("Inference for {} initial conditions".format(n_ics))

    try:
      autoregressive_inference_filetag = params["inference_file_tag"]
    except:
      autoregressive_inference_filetag = ""
    if params.interp > 0:
        autoregressive_inference_filetag = "_coarse"

    # get data and models
    backbone_valid_data_full, finetune_valid_data_0p25_full, finetune_valid_data_0p08_full, backbone, total_model = setup(params)

    # run autoregressive inference for multiple initial conditions
    save_path = os.path.join(args.exp_dir, 'autoregressive_predictions' + autoregressive_inference_filetag+ '.h5')
    for i, ic_ in enumerate(ics):
        logging.info("Initial condition {} of {}".format(i+1, n_ics))
        res = autoregressive_inference(params, ic_, backbone_valid_data_full, finetune_valid_data_0p25_full, finetune_valid_data_0p08_full, backbone, total_model)
        backbone_sr         = res[0]
        backbone_sp         = res[1]
        backbone_vl         = res[2]
        backbone_a          = res[3]
        backbone_au         = res[4]
        backbone_vc         = res[5]
        backbone_ac         = res[6]
        backbone_acu        = res[7]
        finetune_sr         = res[8]
        finetune_sp         = res[9]
        finetune_rmse_w     = res[10]
        finetune_rmse_uw    = res[11]
        finetune_acc_w      = res[12]
        finetune_acc_uw     = res[13]
        finetune_ac         = res[14]
        finetune_acu        = res[15]

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
                                 dtype =np.float32)
                f.create_dataset("backbone_rmse_coarse",     
                                 data=backbone_vc, 
                                 maxshape=[None, args.prediction_length, params.backbone_n_out_channels], 
                                 dtype =np.float32)
                f.create_dataset("backbone_acc",             
                                 data=backbone_a, 
                                 maxshape=[None, args.prediction_length, params.backbone_n_out_channels], 
                                 dtype =np.float32)
                f.create_dataset("backbone_acc_unweighted",  
                                 data=backbone_au, 
                                 maxshape=[None, args.prediction_length, params.backbone_n_out_channels], 
                                 dtype =np.float32)
                f.create_dataset("backbone_acc_coarse",      
                                 data=backbone_ac, 
                                 maxshape=[None, args.prediction_length, params.backbone_n_out_channels], 
                                 dtype =np.float32)
                f.create_dataset("backbone_acc_coarse_unweighted", 
                                 data=backbone_acu, 
                                 maxshape=[None, args.prediction_length, params.backbone_n_out_channels], 
                                 dtype =np.float32)

            f.create_dataset("finetune_ground_truth",
                             data=finetune_sr, 
                             maxshape=[None, args.prediction_length, params.finetune_n_out_channels, 
                                       params.finetune_target_size_h, params.finetune_target_size_w], 
                             dtype=np.float32)
            f.create_dataset("finetune_predicted",       
                             data=finetune_sp, 
                             maxshape=[None, args.prediction_length, params.finetune_n_out_channels, 
                                       params.finetune_target_size_h, params.finetune_target_size_w], 
                             dtype=np.float32)
            f.create_dataset("finetune_input",
                             data=backbone_sp[:,:,[0,15,30,45,60],160:280,490:720], 
                             maxshape=[None, args.prediction_length, params.finetune_n_in_channels, 
                                       params.finetune_in_size_h, params.finetune_in_size_w], 
                             dtype=np.float32)
            f.create_dataset("finetune_rmse_weighted",            
                             data=finetune_rmse_w, 
                             maxshape=[None, args.prediction_length, params.finetune_n_out_channels], 
                             dtype=np.float32)
            f.create_dataset("finetune_rmse_unweighted",            
                             data=finetune_rmse_uw, 
                             maxshape=[None, args.prediction_length, params.finetune_n_out_channels], 
                             dtype=np.float32)
            f.create_dataset("finetune_acc_weighted",             
                             data=finetune_acc_w, 
                             maxshape=[None, args.prediction_length, params.finetune_n_out_channels], 
                             dtype=np.float32)
            f.create_dataset("finetune_acc_unweighted",  
                             data=finetune_acc_uw, 
                             maxshape=[None, args.prediction_length, params.finetune_n_out_channels], 
                             dtype=np.float32)
            f.create_dataset("finetune_acc_coarse_weighted",      
                             data=finetune_ac, 
                             maxshape=[None, args.prediction_length, params.finetune_n_out_channels], 
                             dtype=np.float32)
            f.create_dataset("finetune_acc_coarse_unweighted", 
                             data=finetune_acu, 
                             maxshape=[None, args.prediction_length, params.finetune_n_out_channels], 
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

            f["finetune_ground_truth"].resize((f["finetune_ground_truth"].shape[0] + 1), axis = 0)
            f["finetune_ground_truth"][-1:] = finetune_sr 

            f["finetune_predicted"].resize((f["finetune_predicted"].shape[0] + 1), axis = 0)
            f["finetune_predicted"][-1:] = finetune_sp 

            f["finetune_input"].resize((f["finetune_input"].shape[0] + 1), axis = 0)
            f["finetune_input"][-1:] = backbone_sp[:,:,[0,15,30,45,60],160:280,490:720]

            f["finetune_rmse_weighted"].resize((f["finetune_rmse_weighted"].shape[0] + 1), axis = 0)
            f["finetune_rmse_weighted"][-1:] = finetune_rmse_w

            f["finetune_rmse_unweighted"].resize((f["finetune_rmse_unweighted"].shape[0] + 1), axis = 0)
            f["finetune_rmse_unweighted"][-1:] = finetune_rmse_uw

            f["finetune_acc_weighted"].resize((f["finetune_acc_weighted"].shape[0] + 1), axis = 0)
            f["finetune_acc_weighted"][-1:] = finetune_acc_w

            f["finetune_acc_coarse_weighted"].resize((f["finetune_acc_coarse_weighted"].shape[0] + 1), axis = 0)
            f["finetune_acc_coarse_weighted"][-1:] = finetune_acc_uw

            f["finetune_acc_unweighted"].resize((f["finetune_acc_unweighted"].shape[0] + 1), axis = 0)
            f["finetune_acc_unweighted"][-1:] = finetune_ac

            f["finetune_acc_coarse_unweighted"].resize((f["finetune_acc_coarse_unweighted"].shape[0] + 1), axis = 0)
            f["finetune_acc_coarse_unweighted"][-1:] = finetune_acu

