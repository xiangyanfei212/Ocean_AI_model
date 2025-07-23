"""
BSD 3-Clause License

Copyright (c) 2022, FourCastNet authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

-------------------------------------------------------------------------------

Modifications by: Yanfei Xiang
Copyright (c) 2025, Yanfei Xiang
All rights reserved.

Description of Modifications:
1. Updated to support ocean simulations
2. Added support for autoregressive inference with additional configurations.
3. Integrated land masking during inference.
4. Expanded configuration handling for ocean datasets, normalization schemes, and new model architecture.

"""


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
from icecream import ic
from datetime import datetime
from collections import OrderedDict
from numpy.core.numeric import False_

import torch
import torchvision
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
from torchvision.utils import save_image
from torch.nn.parallel import DistributedDataParallel


sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
from utils.YParams import YParams
from utils.data_loader_multifiles import get_data_loader
from utils.weighted_acc_rmse import weighted_rmse_torch_channels, weighted_acc_torch_channels, unweighted_acc_torch_channels, unweighted_rmse_torch_channels

from utils import logging_utils
from utils import time_utils 
logging_utils.config_logger()

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
    valid_data_loader, valid_dataset = get_data_loader(
        params, params.test_data_path, dist.is_initialized(), train=False)

    img_shape_x = valid_dataset.img_shape_x
    img_shape_y = valid_dataset.img_shape_y
    params.img_shape_x = img_shape_x
    params.img_shape_y = img_shape_y

    in_channels = np.array(params.in_channels)
    out_channels = np.array(params.out_channels)
    n_in_channels = len(in_channels)
    n_out_channels = len(out_channels)

    if params["orography"]:
        params['N_in_channels'] = n_in_channels + 1
    else:
        params['N_in_channels'] = n_in_channels
    params['N_out_channels'] = n_out_channels

    if params.normalization == 'zscore': 
        params.means = np.load(params.global_means_path)
        params.stds = np.load(params.global_stds_path)
    if params.normalization == 'minmax': 
        params.mins = np.load(params.global_mins_path)
        params.maxs = np.load(params.global_maxs_path)

    if params.nettype == 'afno':
        from networks.afnonet import AFNONet as model
    elif params.nettype == 'Masked_AE_Ocean':
        from networks.Masked_AE_Ocean import Masked_Ocean as model
    elif params.nettype == 'Masked_AE_fusion':
        from networks.Masked_AE_fusion import Masked_AFNO as model
    else:
        raise Exception("not implemented")

    checkpoint_file  = params['best_checkpoint_path']
    logging.info('Loading trained model checkpoint from {}'.format(checkpoint_file))
    model = model(params).to(device) 
    model = load_model(model, params, checkpoint_file)
    model = model.to(device)

    files_paths = glob.glob(params.test_data_path + "/*.h5")
    files_paths.sort()

    # which year
    yr = 0
    logging.info('Loading inference data')
    logging.info('Inference data from {}'.format(files_paths[yr]))
    valid_data_full = h5py.File(files_paths[yr], 'r')['fields']

    return valid_data_full, model

    
def autoregressive_inference(params, init_condition, valid_data_full, model): 
    icd = int(init_condition) 
    
    # initialize global variables
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    exp_dir = params['experiment_dir'] 
    dt                = int(params.dt)
    prediction_length = int(params.prediction_length/dt)
    n_history      = params.n_history
    img_shape_x    = params.img_shape_x
    img_shape_y    = params.img_shape_y
    in_channels    = np.array(params.in_channels)
    out_channels   = np.array(params.out_channels)
    atmos_channels = np.array(params.atmos_channels)
    n_in_channels  = len(in_channels)
    n_out_channels = len(out_channels)

    # initialize memory for image sequences and RMSE/ACC
    rmse            = torch.zeros((prediction_length, n_out_channels))
    rmse_unweighted = torch.zeros((prediction_length, n_out_channels))
    acc             = torch.zeros((prediction_length, n_out_channels))
    acc_unweighted  = torch.zeros((prediction_length, n_out_channels))
    seq_real        = torch.zeros((prediction_length, n_out_channels, img_shape_x, img_shape_y))
    seq_pred        = torch.zeros((prediction_length, n_out_channels, img_shape_x, img_shape_y))

    # extract valid data 
    valid_day = np.arange(0, 365)[icd:(icd+prediction_length*dt+n_history*dt):dt]
    logging.info(f'valid_day: {valid_day}')
    valid_date = [time_utils.get_date(params.year, day) for day in valid_day]
    logging.info(f'valid_date: {valid_date}')

    valid_data = valid_data_full[icd:(icd+prediction_length*dt+n_history*dt):dt][:, params.in_channels][:,:,0:720]
    logging.info(f'valid_data_full: {valid_data_full.shape}')
    logging.info(f'valid_data: {valid_data.shape}')
    
    # normalize
    if params.normalization == 'zscore': 
        valid_data = (valid_data - params.means[:,params.in_channels])/params.stds[:,params.in_channels]
    valid_data = torch.as_tensor(valid_data)

    # climatology
    clim = torch.as_tensor(np.load(params.time_means_path))
    clim = clim[:, params.out_channels][:,:,0:720,:]
    clim = (clim - params.means[:, params.out_channels])/params.stds[:, params.out_channels]
    logging.info(f'clim: {clim.shape}')

    # rography
    if params.orography and params.normalization == 'zscore': 
        orography_path = params.orography_norm_zscore_path
    if params.orography:
        orog = torch.as_tensor(np.expand_dims(np.expand_dims(h5py.File(orography_path, 'r')['orog'][0:720], axis = 0), axis = 0))
        logging.info("orography loaded; shape:{}".format(orog.shape))

    # autoregressive inference
    logging.info('Begin autoregressive inference')
    
    if params.fusion_3d_2d:
        with torch.no_grad():
            for i in range(valid_data.shape[0]): 
                if i==0: # start of sequence, t0 --> t0'
                    first = valid_data[0:n_history+1]

                    future = valid_data[n_history+1]

                    for h in range(n_history+1):
                        seq_real[h] = first[h*n_in_channels : (h+1)*n_in_channels, 0:n_out_channels] # extract history from 1st 
                        seq_pred[h] = seq_real[h]
                    if params.perturb:
                        first_3d = gaussian_perturb(first_3d, level=params.n_level, device=device) # perturb the ic
                        first_2d = gaussian_perturb(first_2d, level=params.n_level, device=device) # perturb the ic
                    if params.orography:
                        first = first.to(device, dtype=torch.float)
                        orog  = orog.to(device, dtype=torch.float)
                        first = torch.cat((first, orog), axis=1)
                        print('first:', first.shape)

                        first_3d_t = first[:,:6,:,:] 
                        print('first_3d_t:', first_3d_t.shape)
                        first_3d_s = first[:,6:12,:,:] 
                        first_3d_u = first[:,12:18,:,:] 
                        first_3d_v = first[:,18:24,:,:] 
                        print('first_3d_t:', first_3d_t.shape)
                        first_3d = torch.cat([first_3d_t, first_3d_s, 
                                              first_3d_u, first_3d_v], axis=0)
                        first_3d = first_3d.permute(0,2,3,1)
                        print('first_3d:', first_3d.shape)
                        first_2d = first[:,24:,:,:] 
                        print('first_2d:', first_2d.shape)
                        del first
                        first_3d = first_3d.unsqueeze(0)
                        future_pred = model(first_2d, first_3d, mask_ratio=0)
                    else: # TODO: this function has not been debugged
                        first = first.to(device, dtype=torch.float)
                        future_pred = model(first)

                else: # (t1) --> (t+1)', (t+1)' --> (t+2)', (t+2)' --> (t+3)' ....
                    if i < prediction_length-1:
                        future = valid_data[n_history+i+1]

                    if params.orography:
                        future_pred = future_pred.to(device, dtype=torch.float)
                        future_force = torch.unsqueeze(future[n_out_channels:], dim=0).to(device, dtype=torch.float)
                        orog  = orog.to(device, dtype=torch.float)
                        inp = torch.cat((future_pred, future_force, orog), axis=1) 
                        inp_3d_t = inp[:,:6,:,:] 
                        inp_3d_s = inp[:,6:12,:,:] 
                        inp_3d_u = inp[:,12:18,:,:] 
                        inp_3d_v = inp[:,18:24,:,:] 
                        inp_3d = torch.cat([inp_3d_t, inp_3d_s, inp_3d_u, inp_3d_v], axis=0)
                        inp_3d = inp_3d.permute(0,2,3,1)
                        inp_2d = inp[:,24:,:,:] 

                        inp_3d = inp_3d.unsqueeze(0)

                        future_pred = model(inp_2d, inp_3d, mask_ratio=0)

                    else: # TODO: this function has not been debugged
                        future_force = future[n_out_channels:].to(device, dtype=torch.float)
                        future_pred = model(torch.cat((future_pred, future_force), axis=1)) #autoregressive step

                if i < prediction_length - 1: # not on the last step
                    seq_pred[n_history+i+1] = future_pred.cpu()
                    seq_real[n_history+i+1] = future[:n_out_channels]
                    history_stack = seq_pred[i+1:i+2+n_history]

                future_pred = history_stack

                pred = torch.unsqueeze(seq_pred[i], 0)
                tar  = torch.unsqueeze(seq_real[i], 0)

                # 0:land, 1:ocean
                with h5py.File(params.land_mask_path, 'r') as _f: 
                    mask_data = torch.as_tensor(_f['fields'][:], dtype=bool)
                pred = torch.masked_fill(input=pred, mask=~mask_data, value=0)
                tar  = torch.masked_fill(input=tar,  mask=~mask_data, value=0)

                # Compute metrics 
                if params.normalization == 'zscore': 
                    rmse[i]            = weighted_rmse_torch_channels(pred, tar) * params.stds[:,:n_out_channels,0,0]
                    rmse_unweighted[i] = unweighted_rmse_torch_channels(pred, tar) * params.stds[:,:n_out_channels,0,0]

                acc[i]            = weighted_acc_torch_channels(pred-clim, tar-clim)
                acc_unweighted[i] = unweighted_acc_torch_channels(pred-clim, tar-clim)

    else:
        with torch.no_grad():
            for i in range(valid_data.shape[0]): 
                if i==0: # start of sequence, t0 --> t0'
                    first = valid_data[0:n_history+1]
                    ic(valid_data.shape, first.shape)
                    future = valid_data[n_history+1]
                    ic(future.shape)

                    for h in range(n_history+1):
                        # seq_real[h] = first[h*n_in_channels : (h+1)*n_in_channels, 0:n_out_channels] # extract history from 1st 
                        seq_real[h] = first[h*n_in_channels : (h+1)*n_in_channels, out_channels] # extract history from 1st 
                        seq_pred[h] = seq_real[h]

                    if params.perturb:
                        first = gaussian_perturb(first, level=params.n_level, device=device) # perturb the ic

                    if params.orography:
                        first = first.to(device, dtype=torch.float)
                        orog  = orog.to(device, dtype=torch.float)
                        future_pred = model(torch.cat((first, orog), axis=1))
                    else:
                        first = first.to(device, dtype=torch.float)
                        future_pred = model(first)

                else: # (t1) --> (t+1)', (t+1)' --> (t+2)', (t+2)' --> (t+3)' ....
                    if i < prediction_length-1:
                        future = valid_data[n_history+i+1]

                    if params.orography:
                        future_pred = future_pred.to(device, dtype=torch.float)
                        future_force = torch.unsqueeze(future[atmos_channels], dim=0).to(device, dtype=torch.float)
                        orog  = orog.to(device, dtype=torch.float)

                        inf_one_step_start = time.time()

                        future_pred = model(torch.cat((future_pred, future_force, orog), axis=1)) #autoregressive step

                        inf_one_step_time = time.time() - inf_one_step_start

                    else:
                        future_force = future[atmos_channels].to(device, dtype=torch.float)
                        inf_one_step_start = time.time()
                        future_pred = model(torch.cat((future_pred, future_force), axis=1)) #autoregressive step
                        inf_one_step_time = time.time() - inf_one_step_start

                    logging.info(f'inference one step time: {inf_one_step_time}')

                if i < prediction_length - 1: # not on the last step
                    seq_pred[n_history+i+1] = future_pred.cpu()
                    seq_real[n_history+i+1] = future[out_channels]
                    history_stack = seq_pred[i+1:i+2+n_history]

                future_pred = history_stack

                pred = torch.unsqueeze(seq_pred[i], 0)
                tar  = torch.unsqueeze(seq_real[i], 0)

                with h5py.File(params.land_mask_path, 'r') as _f: 
                    mask_data = torch.as_tensor(_f['fields'][:,out_channels], dtype=bool)
                    # ic(mask_data.shape, pred.shape, tar.shape)
                pred = torch.masked_fill(input=pred, mask=~mask_data, value=0)
                tar  = torch.masked_fill(input=tar,  mask=~mask_data, value=0)

                # Compute metrics 
                if params.normalization == 'zscore': 
                    rmse[i]            = weighted_rmse_torch_channels(pred, tar) * params.stds[:,out_channels,0,0]
                    rmse_unweighted[i] = unweighted_rmse_torch_channels(pred, tar) * params.stds[:,out_channels,0,0]

                acc[i]            = weighted_acc_torch_channels(pred-clim, tar-clim)
                acc_unweighted[i] = unweighted_acc_torch_channels(pred-clim, tar-clim)

                # logging.info('Predicted timestep {} of {}. {} RMSE: {}, ACC: {}'.format(i, prediction_length,  rmse[i], acc[i]))

    seq_real = seq_real.numpy()
    seq_pred = seq_pred.numpy()
    rmse = rmse.numpy()
    rmse_unweighted = rmse_unweighted.numpy()
    acc = acc.numpy()
    acc_unweighted = acc_unweighted.numpy()

    return (np.expand_dims(seq_real[n_history:], 0), # no mask
            np.expand_dims(seq_pred[n_history:], 0), # no mask 
            np.expand_dims(rmse,0),                 # mask land 
            np.expand_dims(rmse_unweighted,0),      # mask land
            np.expand_dims(acc, 0),                 # mask land
            np.expand_dims(acc_unweighted, 0))      # mask land


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", default='../exp_15_levels', type=str)
    parser.add_argument("--config", default='full_field', type=str)
    parser.add_argument("--run_num", default='00', type=str)
    # parser.add_argument("--yaml_config", default='../config/AFNO.yaml', type=str)
    parser.add_argument("--prediction_length", default=30, type=int)
    parser.add_argument("--decorrelation_time", default=30, type=int)
    parser.add_argument("--n_samples_per_year", default=365, type=int)
    parser.add_argument("--finetune_dir", default='', type=str)

    parser.add_argument("--ics_type", default='default', type=str)
    parser.add_argument("--year", default=2012, type=int)
    parser.add_argument("--date_strings", default='01/01/2021 00:00:00,01/02/2021 00:00:00,01/03/2021 00:00:00', type=str)
    
    args = parser.parse_args()
    if args.finetune_dir == '':
        expDir = os.path.join(args.exp_dir, args.config, str(args.run_num))
    else:
        expDir = os.path.join(args.exp_dir, args.config, str(args.run_num), args.finetune_dir)
    logging.info(f'expDir: {expDir}')
    config_path = os.path.join(expDir, 'config.yaml')
    params = YParams(config_path, args.config)
    params['experiment_dir']       = expDir 
    params['best_checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')

    params['resuming']           = False
    params['interp']             = 0 
    params['world_size']         = 1
    params['local_rank']         = 0
    params['global_batch_size']  = params.batch_size
    params['prediction_length']  = args.prediction_length
    params['decorrelation_time'] = args.decorrelation_time
    params['n_samples_per_year'] = args.n_samples_per_year
    params['multi_steps_finetune'] = 1
    params['year']         = args.year
    params['ics_type']     = args.ics_type
    params['date_strings'] = args.date_strings.split(",")

    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True

    # Set up directory
    # params['best_checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/ckpt.tar')

    # set up logging
    logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'inference.log'))
    logging_utils.log_versions()
    params.log()

    if params["ics_type"] == 'default':
        num_samples = params.n_samples_per_year-params.prediction_length
        stop = num_samples
        ics = np.arange(0, stop, params.decorrelation_time)
        n_ics = len(ics)
        print('init_condition:', ics)

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
            for ii in range(n_ics):
                ics.append(int(day_of_year))
        else:
            for date in date_strings:
                date_obj = datetime.strptime(date,'%d/%m/%Y-%H:%M:%S') 
                day_of_year = date_obj.timetuple().tm_yday - 1
                logging.info(f"date: {date}, day_of_year:{day_of_year}")
                ics.append(day_of_year)
        n_ics = len(ics)
    logging.info("Inference for {} initial conditions".format(n_ics))

    try:
      autoregressive_inference_filetag = params["inference_file_tag"]
    except:
      autoregressive_inference_filetag = ""
    if params.interp > 0:
        autoregressive_inference_filetag = "_coarse"

    # get data and models
    valid_data_full, model = setup(params)

    # initialize lists for image sequences and RMSE/ACC
    rmse = []
    rmse_unweighted = []

    acc = []
    acc_unweighted = []

    seq_pred = []
    seq_real = []

    # run autoregressive inference for multiple initial conditions
    for i, ic_ in enumerate(ics):
        logging.info("Initial condition {} of {}".format(i+1, n_ics))
        seq_real, seq_pred, rmse, rmse_unweighted, acc, acc_unweighted = autoregressive_inference(params, ic_, valid_data_full, model)

        prediction_length = seq_real[0].shape[0]
        n_out_channels = seq_real[0].shape[1]
        img_shape_x = seq_real[0].shape[2]
        img_shape_y = seq_real[0].shape[3]

        # save predictions and loss
        save_path = os.path.join(
            params['experiment_dir'],
            'autoregressive_predictions' + autoregressive_inference_filetag+ '_' + str(params.year) +'.h5')
        logging.info("Saving to {}".format(save_path))
        print(f'saving to {save_path}')
        if i==0 or len(rmse) == 0:
            f = h5py.File(save_path, 'w')
            f.create_dataset(
                    "year",
                    data=args.year, 
                    dtype=np.int32)
            f.create_dataset(
                    "ground_truth",
                    data=seq_real,
                    maxshape=[None, prediction_length, n_out_channels, img_shape_x, img_shape_y], 
                    dtype=np.float32)
            f.create_dataset(
                    "init_day_of_year",
                    data=[ic_], 
                    maxshape=[None], 
                    dtype=np.int32)
            f.create_dataset(
                    "predicted",       
                    data=seq_pred, 
                    maxshape=[None, prediction_length, n_out_channels, img_shape_x, img_shape_y], 
                    dtype=np.float32)
            f.create_dataset(
                    "rmse",            
                    data=rmse, 
                    maxshape=[None, prediction_length, n_out_channels], 
                    dtype =np.float32)
            f.create_dataset(
                    "rmse_unweighted", 
                    data=rmse_unweighted, 
                    maxshape=[None, prediction_length, n_out_channels], 
                    dtype =np.float32)
            f.create_dataset(
                    "acc",             
                    data=acc, 
                    maxshape=[None, prediction_length, n_out_channels], 
                    dtype =np.float32)
            f.create_dataset(
                    "acc_unweighted",  
                    data=acc_unweighted, 
                    maxshape=[None, prediction_length, n_out_channels], 
                    dtype =np.float32)
            f.close()
        else:
            f = h5py.File(save_path, 'a')
            f["init_day_of_year"].resize((f["init_day_of_year"].shape[0] + 1), axis = 0)
            f["init_day_of_year"][-1:] = ic_

            f["ground_truth"].resize((f["ground_truth"].shape[0] + 1), axis = 0)
            f["ground_truth"][-1:] = seq_real 

            f["predicted"].resize((f["predicted"].shape[0] + 1), axis = 0)
            f["predicted"][-1:] = seq_pred 

            f["rmse"].resize((f["rmse"].shape[0] + 1), axis = 0)
            f["rmse"][-1:] = rmse 

            f["rmse_unweighted"].resize((f["rmse_unweighted"].shape[0] + 1), axis = 0)
            f["rmse_unweighted"][-1:] = rmse_unweighted 

            f["acc"].resize((f["acc"].shape[0] + 1), axis = 0)
            f["acc"][-1:] = acc

            f["acc_unweighted"].resize((f["acc_unweighted"].shape[0] + 1), axis = 0)
            f["acc_unweighted"][-1:] = acc_unweighted

            f.close()

