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

Description:
This script is a heavily modified version of the original FourCastNet code.
The modifications are tailored for fine-tuning oceanographic data models with
distributed training support, mixed precision, and advanced loss functions.

Key Modifications:
1. Reorganized the training loop and added support for downstream models with frozen backbones.
2. Integrated multiple downstream configurations, including WaveNet and DownScalingNet.
3. Added momentum conservation loss and land masking for downstream data.
4. Enhanced logging and experiment tracking with Weights & Biases (WandB).
5. Refactored data loaders to handle separate backbone and downstream datasets.
"""

import os
import sys
import time
import h5py
import json
import torch
import wandb
import pickle
import logging
import argparse
import cProfile
import numpy as np
import matplotlib.pyplot as plt
from apex import optimizers
from shutil import copyfile
from collections import OrderedDict

import torchvision
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
from torchsummary import summary
from torchvision.utils import save_image
from torch.nn.parallel import DistributedDataParallel

from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams
from utils.darcy_loss import LpLoss, Momentum_Conservation
from utils.data_loader_multifiles import get_downstream_data_loader
from utils.weighted_acc_rmse import weighted_acc, weighted_rmse, weighted_rmse_torch

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict

from networks.Masked_AE_Ocean import Masked_Ocean

os.environ["WANDB_API_KEY"] = "" # Your wandb API KEY
os.environ["WANDB_MODE"] = "dryrun" # optional: dryrun, offline
# if 'WANDB_MODE' is set to 'dryrun', first 'cd wandb', then run 'wandb sync + $dir', the log will updated to the wandb website.

class Trainer():
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad) 

    def __init__(self, params, world_rank):

        self.params = params
        self.world_rank = world_rank
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

        # Init wandb
        if params.log_to_wandb:
            wandb.init(config  = params, 
                       name    = params.name, 
                       group   = params.group, 
                       project = params.project,
                       entity  = params.entity)

        # Init gpu
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        self.device = torch.device('cuda', local_rank)
        logging.info('device: %s' % self.device)

        # Load data
        logging.info('rank %d, begin data loader init' % world_rank)
        self.train_data_loader, self.train_dataset, self.train_sampler = get_downstream_data_loader(
                    params, 
                    params.train_data_path, # backbone data 
                    params.downstream_train_data_path, # downstream data
                    dist.is_initialized(),
                    train=True)
        self.valid_data_loader, self.valid_dataset, self.valid_sampler = get_downstream_data_loader(
                    params, 
                    params.valid_data_path, # backbone data
                    params.downstream_valid_data_path, # downstream data
                    dist.is_initialized(), 
                    train=True)
        logging.info('rank %d, data loader initialized' % world_rank)

        # Loss function
        self.loss_obj = LpLoss()
        self.mom_loss = Momentum_Conservation()

        # Load model
        if params.downstream_config == 'DownScalingNet': 
            from downstream_networks.DownScalingNet import DownScalingNet as downstream_model

        if params.downstream_config == 'WaveNet': 
            from downstream_networks.WaveNet import WaveNet as downstream_model

        if params.downstream_config == 'BiochemicalNet': 
            from downstream_networks.BiochemicalNet import BiochemicalNet as downstream_model


        # init model, including the freezed backbone and un-freezed downstream model
        backbone = Masked_Ocean(params)
        # load pretrained params to backbone
        logging.info("Starting from pretrained one-step model at %s"%params.pretrained_ckpt_path)
        backbone = self.restore_checkpoint(backbone, params.pretrained_ckpt_path)

        if self.params.freeze_backbone == 1:
            logging.info("Freeze the backbone")
            self.switch_off_grad(backbone) # freeze backbone

        logging.info('Init the downstream model')
        self.model = downstream_model(backbone, params).to(self.device)

        # print('---------------------------------------')
        # for name, param in self.model.named_parameters():
        #     print(name, param.grad, param.requires_grad)
        # print('---------------------------------------')

        if params.log_to_wandb:
            wandb.watch(self.model)

        # init optimizer
        if params.optimizer_type == 'FusedAdam':
            self.optimizer = optimizers.FusedAdam(self.model.parameters(), lr=params.lr)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr)

        if params.enable_amp == True:
            self.gscaler = amp.GradScaler()

        # distribution
        if dist.is_initialized():
            self.model = DistributedDataParallel(self.model,
                                                 device_ids    = [params.local_rank],
                                                 output_device = [params.local_rank],
                                                 find_unused_parameters = True)

        self.iters = 0
        self.startEpoch = 0
        self.epoch = self.startEpoch
        logging.info("Adding %d epochs specified in config file for refining pretrained model"%params.downstream_max_epochs)

        # Dynamical Learning rate
        if params.scheduler == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                                        factor=0.2, 
                                                                        patience=5, 
                                                                        mode='min')
        elif params.scheduler == 'CosineAnnealingLR': 
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                                                                        T_max=params.downstream_max_epochs,
                                                                        last_epoch=self.startEpoch - 1)
        else:
            self.scheduler = None

        if params.log_to_screen:
            logging.info("Number of trainable model parameters: {}".format(self.count_parameters()))

    def switch_off_grad(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def train(self):
        if self.params.log_to_screen:
            logging.info("Starting Training Loop...")

        best_valid_loss = 1.e6
        for epoch in range(self.startEpoch, self.params.downstream_max_epochs):
            if dist.is_initialized():
                self.train_sampler.set_epoch(epoch)
                self.valid_sampler.set_epoch(epoch)

            start = time.time()
            tr_time, data_time, step_time, train_logs = self.train_one_epoch() 
            valid_time, valid_logs = self.validate_one_epoch()

            if self.params.scheduler == 'ReduceLROnPlateau':
                self.scheduler.step(valid_logs['valid_loss'])
            elif self.params.scheduler == 'CosineAnnealingLR':
                self.scheduler.step()
                if self.epoch >= self.params.downstream_max_epochs:
                    logging.info("Terminating training after reaching params.downstream_max_epochs while LR scheduler is set to CosineAnnealingLR")
                    exit()

            if self.params.log_to_wandb:
                for pg in self.optimizer.param_groups:
                    lr = pg['lr']
                wandb.log({'lr': lr})

            if self.world_rank == 0:
                if self.params.save_checkpoint:
                    # checkpoint at the end of every epoch
                    self.save_checkpoint(self.params.checkpoint_path)
                    if valid_logs['valid_loss_l2'] <= best_valid_loss:
                        logging.info('Val loss0 improved from {} to {}'.format(best_valid_loss, valid_logs['valid_loss_l2']))
                        self.save_checkpoint(self.params.best_checkpoint_path)
                        best_valid_loss = valid_logs['valid_loss_l2']

            if self.params.log_to_screen:
                logging.info('Time taken for epoch {} is {} sec'.format(epoch+1, time.time() - start))
                logging.info('train data time={}, train per epoch time={}, train per step time={}, valid time={}'.format(data_time, tr_time, step_time, valid_time))

                if params.downstream_config == 'DownScalingNet':
                    logging.info('Train loss total: {}. Train loss mom: {}. Train loss l2: {}'.format(train_logs['train_loss_tot'], train_logs['train_loss_mom'], train_logs['train_loss_l2']))
                    logging.info('Valid loss total: {}. Valid loss mom: {}. Valid loss l2: {}'.format(valid_logs['valid_loss_tot'], valid_logs['valid_loss_mom'], valid_logs['valid_loss_l2']))
                else:
                    logging.info('Train loss l2: {}'.format(train_logs['train_loss_l2']))
                    logging.info('Valid loss l2: {}. Valid loss l1: {}'.format(valid_logs['valid_loss_l2'], valid_logs['valid_loss_l1']))


    def train_one_epoch(self):
        self.epoch += 1
        tr_time = 0
        data_time = 0
        self.model.train()

        steps_in_one_epoch = 0
        for i, data in enumerate(self.train_data_loader, 0):
            self.iters += 1
            steps_in_one_epoch += 1 

            data_start = time.time()
            backbone_inp, backbone_tar, downstream_inp, downstream_tar = map(lambda x: x.to(self.device, dtype=torch.float), data)
            # print(f'backbone_inp: {backbone_inp.shape}, backbone_tar:{backbone_tar.shape}, downstream_inp: {downstream_inp.shape}, downstream_tar:{downstream_tar.shape}')
            if downstream_tar.dim() == 3:
                downstream_tar = torch.unsqueeze(downstream_tar, 1)
            data_time += time.time() - data_start

            tr_start = time.time()
            self.model.zero_grad()
            with amp.autocast(self.params.enable_amp):
                backbone_gen, downstream_gen = self.model(backbone_inp, downstream_inp)
                # print(f'downstream_gen: {downstream_gen.shape}, backbone_gen: {backbone_gen.shape}')

                # land mask
                if self.params.land_mask:
                    # 0:land, 1:ocean
                    with h5py.File(self.params.downstream_land_mask_path, 'r') as _f: 
                        # logging.info(f"Loading land mask data from {self.params.land_mask_path}")
                        if params.downstream_config == 'WaveNet':
                            mask_data = torch.as_tensor(_f['fields'][:360,:]).to(self.device, dtype=torch.bool)
                        elif params.downstream_config == 'DownScalingNet':
                            mask_data = torch.as_tensor(_f['fields']).to(self.device, dtype=torch.bool)
                            mask_data = mask_data[:,params.downstream_out_channels,:,:]
                        else:
                            mask_data = torch.as_tensor(_f['fields']).to(self.device, dtype=torch.bool)
                    downstream_gen = torch.masked_fill(input=downstream_gen, mask=~mask_data, value=0)
                    downstream_tar = torch.masked_fill(input=downstream_tar, mask=~mask_data, value=0)

                # L2 loss
                l2_loss = self.loss_obj(downstream_gen, downstream_tar)

                # loss for momentum conservation
                if params.downstream_config == 'DownScalingNet':
                    with h5py.File(self.params.downstream_topo_0p08_path, 'r') as _f:
                        topo = torch.as_tensor(_f['Band1_zscore']).to(self.device)
                    mom_loss = self.mom_loss(topo, downstream_gen, downstream_tar)

                if params.use_mom_loss == 1:
                    loss = l2_loss + mom_loss
                else:
                    loss = l2_loss
                
            if self.params.enable_amp:
                self.gscaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10, norm_type=2)
                self.gscaler.step(self.optimizer)
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10, norm_type=2)
                self.optimizer.step()

            if self.params.enable_amp:
                self.gscaler.update()

            tr_time += time.time() - tr_start

        # time of one step in epoch
        step_time = tr_time / steps_in_one_epoch
        if params.downstream_config == 'DownScalingNet':
            logs = {'train_loss_l2': l2_loss, 'train_loss_mom': mom_loss, 'train_loss_tot': loss}
        else:
            logs = {'train_loss_l2': l2_loss}

        if dist.is_initialized():
            for key in sorted(logs.keys()):
                dist.all_reduce(logs[key].detach())
                logs[key] = float(logs[key] / dist.get_world_size())
        
        logs['train_one_epoch_time'] = tr_time
        logs['train_one_step_time'] = step_time
        if self.params.log_to_wandb:
            wandb.log(logs, step=self.epoch)

        return tr_time, data_time, step_time, logs

    def validate_one_epoch(self):

        logging.info('validating...')
        self.model.eval()

        if params.downstream_config == 'DownScalingNet':
            valid_buff     = torch.zeros((5), dtype=torch.float32, device=self.device)
            valid_loss_l2  = valid_buff[0].view(-1) # 0
            valid_loss_mom = valid_buff[1].view(-1) # 0
            valid_loss_tot = valid_buff[2].view(-1) # 0
            valid_loss_l1  = valid_buff[3].view(-1) # 0
            valid_steps    = valid_buff[4].view(-1) # 0
        else:
            valid_buff     = torch.zeros((3), dtype=torch.float32, device=self.device)
            valid_loss_l2  = valid_buff[0].view(-1) # 0
            valid_loss_l1  = valid_buff[1].view(-1) # 0
            valid_steps    = valid_buff[2].view(-1) # 0

        valid_start = time.time()
        sample_idx = np.random.randint(len(self.valid_data_loader))
        with torch.no_grad():
            for i, data in enumerate(self.valid_data_loader, 0):

                backbone_inp, backbone_tar, downstream_inp, downstream_tar = map(lambda x: x.to(self.device, dtype=torch.float), data)
                if downstream_tar.dim() == 3:
                    downstream_tar = torch.unsqueeze(downstream_tar, 1)

                if params.add_noise:
                    downstream_inp = downstream_inp + torch.normal(mean=params.noise_mean, std=params.noise_std, 
                                                               size=downstream_inp.shape, device=self.device)

                backbone_gen, downstream_gen = self.model(backbone_inp, downstream_inp)

                # land mask
                if self.params.land_mask:
                    # 0:land, 1:ocean
                    with h5py.File(self.params.downstream_land_mask_path, 'r') as _f: 
                        # logging.info(f"Loading land mask data from {self.params.land_mask_path}")
                        if params.downstream_config == 'WaveNet':
                            mask_data = torch.as_tensor(_f['fields'][:360,:]).to(self.device, dtype=torch.bool)
                        elif params.downstream_config == 'DownScalingNet':
                            mask_data = torch.as_tensor(_f['fields']).to(self.device, dtype=torch.bool)
                            mask_data = mask_data[:,params.downstream_out_channels,:,:]
                        else:
                            mask_data = torch.as_tensor(_f['fields']).to(self.device, dtype=torch.bool)
                    downstream_gen = torch.masked_fill(input=downstream_gen, mask=~mask_data, value=0)
                    downstream_tar = torch.masked_fill(input=downstream_tar, mask=~mask_data, value=0)

                loss_l2 = self.loss_obj(downstream_gen, downstream_tar)
                valid_loss_l2 += loss_l2

                if params.downstream_config == 'DownScalingNet':
                    with h5py.File(self.params.downstream_topo_0p08_path, 'r') as _f:
                        topo = torch.as_tensor(_f['Band1_zscore']).to(self.device)
                    loss_mom = self.mom_loss(topo, downstream_gen, downstream_tar)
                    valid_loss_mom += loss_mom
                
                    loss_tot = loss_l2 + loss_mom
                    valid_loss_tot += loss_tot

                valid_loss_l1 += nn.functional.l1_loss(downstream_gen, downstream_tar)

                valid_steps += 1.

                # save fields for vis before log norm
                os.makedirs(params['experiment_dir'] + "/" + str(i), exist_ok =True)
                if params.downstream_config == 'DownScalingNet':
                    save_image(torch.cat((downstream_gen[0, 0],
                                          torch.zeros((self.valid_dataset.downstream_0p08_img_shape_x,4)).to(self.device, dtype=torch.float),
                                          downstream_tar[0, 0]), axis=1),
                               os.path.join(params['experiment_dir'], str(i), f'epoch_{self.epoch}.png'))
                else:
                    save_image(torch.cat((downstream_gen[0, 0],
                                          torch.zeros((self.valid_dataset.downstream_img_shape_x,4)).to(self.device, dtype=torch.float),
                                          downstream_tar[0, 0]), axis=1),
                               os.path.join(params['experiment_dir'], str(i), f'epoch_{self.epoch}.png'))

        if dist.is_initialized():
            dist.all_reduce(valid_buff)

        # divide by number of steps
        if params.downstream_config == 'DownScalingNet':
            valid_buff[0:4] = valid_buff[0:4] / valid_buff[4] # loss/steps, l1/steps
            valid_buff_cpu = valid_buff.detach().cpu().numpy()

            valid_time = time.time() - valid_start
            
            logs = { 'valid_loss_l2':   valid_buff_cpu[0],
                     'valid_loss_mom':  valid_buff_cpu[1],
                     'valid_loss_tot':  valid_buff_cpu[2],
                     'valid_loss_l1':   valid_buff_cpu[3]}
        else:
            valid_buff[0:2] = valid_buff[0:2] / valid_buff[2] # loss/steps, l1/steps
            valid_buff_cpu = valid_buff.detach().cpu().numpy()

            valid_time = time.time() - valid_start
            
            logs = { 'valid_loss_l2': valid_buff_cpu[0],
                     'valid_loss_l1': valid_buff_cpu[1]}

        if self.params.log_to_wandb:
            wandb.log(logs, step=self.epoch)

        return valid_time, logs

    def load_model(self, model_path):
        if self.params.log_to_screen:
            logging.info('Loading the model weights from {}'.format(model_path))

        checkpoint = torch.load(model_path, map_location='cuda:{}'.format(self.params.local_rank))

        if dist.is_initialized():
            self.model.load_state_dict(checkpoint['model_state'])
        else:
            new_model_state = OrderedDict()
            model_key = 'model_state' if 'model_state' in checkpoint else 'state_dict'
            for key in checkpoint[model_key].keys():
                if 'module.' in key:  # model was stored using ddp which prepends module
                    name = str(key[7:])
                    new_model_state[name] = checkpoint[model_key][key]
                else:
                    new_model_state[key] = checkpoint[model_key][key]
            self.model.load_state_dict(new_model_state)
            self.model.eval()

    def save_checkpoint(self, checkpoint_path, model=None):
        """ We intentionally require a checkpoint_dir to be passed
            in order to allow Ray Tune to use this function """

        if not model:
            model = self.model

        torch.save({'iters': self.iters, 
                    'epoch': self.epoch, 
                    'model_state': model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, 
                    checkpoint_path)

    def restore_checkpoint(self, model, checkpoint_path):
        """ We intentionally require a checkpoint_dir to be passed
            in order to allow Ray Tune to use this function """
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.params.local_rank))
        try:
            model.load_state_dict(checkpoint['model_state'])
        except:
            new_state_dict = OrderedDict()
            for key, val in checkpoint['model_state'].items():
                name = key[7:]
                new_state_dict[name] = val
            model.load_state_dict(new_state_dict)
        self.iters = checkpoint['iters']
        self.startEpoch = checkpoint['epoch']

        if self.params.resuming:  
        # restore checkpoint is used for finetuning as well as resuming. 
        # If finetuning (i.e., not resuming), restore checkpoint does not load optimizer state, instead uses config specified lr.
           self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default='./config/Model_downstream.yaml', type=str)  
    parser.add_argument("--pretrained_dir", default='', type=str) # the location of backbone's parameters and current experiment directory
    parser.add_argument("--downstream_config", default='WaveNet', type=str)
    parser.add_argument("--freeze_backbone", default=1, type=int)
    parser.add_argument("--downstream_max_epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=16, type=int)

    parser.add_argument("--use_mom_loss", default=0, type=int)

    parser.add_argument("--add_noise", default=0, type=int)
    parser.add_argument("--noise_mean", default=0, type=int)
    parser.add_argument("--noise_std", default=0, type=float)

    parser.add_argument("--wandb_group", default='downstream_WaveNet', type=str)
    parser.add_argument("--run_num", default='', type=str)
    parser.add_argument("--enable_amp", action='store_true')
    parser.add_argument("--epsilon_factor", default=0, type=float)
    parser.add_argument("--local_rank", default=-1, type=int, help='node rank for distributed training')
    args = parser.parse_args()
    print(args)

    params = YParams(os.path.abspath(args.yaml_config), args.downstream_config, False)

    params['downstream_config']      = args.downstream_config
    params['freeze_backbone']      = args.freeze_backbone
    params['epsilon_factor']       = args.epsilon_factor
    params['downstream_max_epochs']  = args.downstream_max_epochs

    params['use_mom_loss']         = args.use_mom_loss

    params['add_noise']            = args.add_noise
    params['noise_mean']           = args.noise_mean
    params['noise_std']            = args.noise_std

    params['enable_amp']           = args.enable_amp  # Automatic Mixed Precision Training
    params['resuming']             = False            # this is not downstream the backbone
    params['pretrained_ckpt_path'] = os.path.join(args.pretrained_dir, 'training_checkpoints/best_ckpt.tar') # backbone parameters

    params['world_size'] = 1
    if 'WORLD_SIZE' in os.environ:
        params['world_size'] = int(os.environ['WORLD_SIZE']) 

    print('Initialize distributed process group...')
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    params['local_rank'] = local_rank 

    torch.backends.cudnn.benchmark = True
    world_rank = dist.get_rank()

    params['global_batch_size'] = args.batch_size
    params['batch_size'] = int(args.batch_size // params['world_size'])  # batch size must be divisible by the number of gpu's

    # current experiment directory
    expDir = os.path.join(args.pretrained_dir, args.downstream_config, args.run_num)

    if world_rank == 0:
        os.makedirs(expDir, exist_ok=True)
        os.makedirs(os.path.join(expDir, 'training_checkpoints/'), exist_ok=True)
        copyfile(os.path.abspath(args.yaml_config), os.path.join(expDir, 'config.yaml'))

    params['experiment_dir']       = os.path.abspath(expDir)
    params['checkpoint_path']      = os.path.join(expDir, 'training_checkpoints/ckpt.tar') 
    params['best_checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')

    # wandb setting 
    params['entity']  = "ocean_ai_model"    # team name 
    params['project'] = "ai4science"        # project name
    params['group']   = args.wandb_group    # group name
    params['name']    = params['nettype'] + '_' + str(args.run_num)  # experiment name

    if world_rank == 0:
        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'out.log'))
        logging_utils.log_versions()
        params.log()

    params['log_to_wandb']  = (world_rank == 0) and params['log_to_wandb']

    params['in_channels']    = np.array(params['in_channels'])
    params['out_channels']   = np.array(params['out_channels'])
    params['N_out_channels'] = len(params['out_channels'])
    if params.orography:
        params['N_in_channels'] = len(params['in_channels']) + 1
    else:
        params['N_in_channels'] = len(params['in_channels']) 

    if world_rank == 0:
        hparams = ruamelDict()
        yaml = YAML()
        for key, value in params.params.items():
            hparams[str(key)] = str(value)
        with open(os.path.join(expDir, 'hyperparams.yaml'), 'w') as hpfile:
            yaml.dump(hparams, hpfile)
    logging.info(f'run_num: {args.run_num}')

    trainer = Trainer(params, world_rank)
    trainer.train()
    logging.info('DONE ---- rank %d' % world_rank)
