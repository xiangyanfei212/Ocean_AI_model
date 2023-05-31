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
from utils.darcy_loss import LpLoss
from utils.data_loader_multifiles import get_data_loader
from utils.weighted_acc_rmse import weighted_acc, weighted_rmse, weighted_rmse_torch

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict


os.environ["WANDB_API_KEY"] = "5aa5d8287afeb6e89eb8100e3fc6dc595e367af3"
#os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_MODE"] = "dryrun"

class Trainer():
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)  # 返回参数个数

    def __init__(self, params, world_rank):

        self.params = params
        self.world_rank = world_rank
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

        # wandb库初始化
        if params.log_to_wandb:
            wandb.init(config=params, 
                       name=params.name, 
                       group=params.group, 
                       project=params.project,
                       entity=params.entity)

        # 加载数据
        logging.info('rank %d, begin data loader init' % world_rank)
        self.train_data_loader, self.train_dataset, self.train_sampler = get_data_loader(params, params.train_data_path,
                                                                                         dist.is_initialized(),
                                                                                         train=True)
        self.valid_data_loader, self.valid_dataset = get_data_loader(params, params.valid_data_path,
                                                                     dist.is_initialized(), train=False)

        self.loss_obj = LpLoss()
        logging.info('rank %d, data loader initialized' % world_rank)

        # 加载模型
        if params.nettype == 'afno':
            from networks.afnonet import AFNONet as model
        elif params.nettype == 'model_v2':
            from networks.model_v2 import Model_iter as model
        else:
            raise Exception("not implemented")
        self.model = model(params).to(self.device)

        if self.params.enable_nhwc:
            # NHWC: Convert model to channels_last memory format
            self.model = self.model.to(memory_format=torch.channels_last)

        if params.log_to_wandb:
            wandb.watch(self.model)

        if params.optimizer_type == 'FusedAdam':
            self.optimizer = optimizers.FusedAdam(self.model.parameters(), lr = params.lr)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = params.lr)

        if params.enable_amp == True:
            self.gscaler = amp.GradScaler()

        if dist.is_initialized():
            self.model = DistributedDataParallel(self.model,
                                                 device_ids=[params.local_rank],
                                                 output_device=[params.local_rank],
                                                 find_unused_parameters=True)

        self.iters = 0
        self.startEpoch = 0
        if params.resuming:
            logging.info("Loading checkpoint %s" % params.checkpoint_path)
            self.restore_checkpoint(params.checkpoint_path)

        self.epoch = self.startEpoch

        # 动态学习率
        if params.scheduler == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.2, 
                                                                        patience=5, mode='min')
        elif params.scheduler == 'CosineAnnealingLR': 
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=params.max_epochs,
                                                                        last_epoch=self.startEpoch - 1)
        else:
            self.scheduler = None

        '''if params.log_to_screen:
          logging.info(self.model)'''
        if params.log_to_screen:
            logging.info("Number of trainable model parameters: {}".format(self.count_parameters()))

    def switch_off_grad(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def train(self):
        if self.params.log_to_screen:
            logging.info("Starting Training Loop...")

        best_valid_loss = 1.e6
        for epoch in range(self.startEpoch, self.params.max_epochs):
            if dist.is_initialized():
                self.train_sampler.set_epoch(epoch)
            #   self.valid_sampler.set_epoch(epoch)

            start = time.time()
            tr_time, data_time, train_logs = self.train_one_epoch() 
            valid_time, valid_logs = self.validate_one_epoch()

            if epoch == self.params.max_epochs - 1 and self.params.prediction_type == 'direct':
                valid_weighted_rmse = self.validate_final()

            if self.params.scheduler == 'ReduceLROnPlateau':
                self.scheduler.step(valid_logs['valid_loss'])
            elif self.params.scheduler == 'CosineAnnealingLR':
                self.scheduler.step()
                if self.epoch >= self.params.max_epochs:
                    logging.info("Terminating training after reaching params.max_epochs while LR scheduler is set to CosineAnnealingLR")
                    exit()

            if self.params.log_to_wandb:
                for pg in self.optimizer.param_groups:
                    lr = pg['lr']
                wandb.log({'lr': lr})

            if self.world_rank == 0:
                if self.params.save_checkpoint:
                    # checkpoint at the end of every epoch
                    self.save_checkpoint(self.params.checkpoint_path)
                    if valid_logs['valid_loss'] <= best_valid_loss:
                        logging.info('Val loss improved from {} to {}'.format(best_valid_loss, valid_logs['valid_loss']))
                        self.save_checkpoint(self.params.best_checkpoint_path)
                        best_valid_loss = valid_logs['valid_loss']

            if self.params.log_to_screen:
                logging.info('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
                logging.info('train data time={}, train step time={}, valid step time={}'.format(data_time, tr_time, valid_time))
                logging.info('Train loss: {}. Valid loss: {}'.format(train_logs['loss'], valid_logs['valid_loss']))

    #        if epoch==self.params.max_epochs-1 and self.params.prediction_type == 'direct':
    #          logging.info('Final Valid RMSE: Z500- {}. T850- {}, 2m_T- {}'.format(valid_weighted_rmse[0], valid_weighted_rmse[1], valid_weighted_rmse[2]))


    def train_one_epoch(self):
        self.epoch += 1
        tr_time = 0
        data_time = 0
        self.model.train()

        for i, data in enumerate(self.train_data_loader, 0):
            self.iters += 1
            # adjust_LR(optimizer, params, iters)
            data_start = time.time()

            inp, tar = map(lambda x: x.to(self.device, dtype=torch.float), data)  
            # logging.info(f'inp: {inp.shape}, tar:{tar.shape}')

            if self.params.enable_nhwc:
                inp = inp.to(memory_format=torch.channels_last)
                tar = tar.to(memory_format=torch.channels_last)

            data_time += time.time() - data_start

            tr_start = time.time()
            self.model.zero_grad()
            with amp.autocast(self.params.enable_amp):
                gen = self.model(inp)
                gen.to(self.device, dtype=torch.float)
                loss_gen = self.loss_obj(gen, tar)
                loss = loss_gen

            if self.params.enable_amp:
                self.gscaler.scale(loss).backward()
                self.gscaler.step(self.optimizer)
            else:
                loss.backward()
                self.optimizer.step()

            if self.params.enable_amp:
                self.gscaler.update()

            tr_time += time.time() - tr_start

        logs = {'loss': loss}

        if dist.is_initialized():
            for key in sorted(logs.keys()):
                dist.all_reduce(logs[key].detach())
                logs[key] = float(logs[key] / dist.get_world_size())

        if self.params.log_to_wandb:
            wandb.log(logs, step=self.epoch)

        return tr_time, data_time, logs

    def validate_one_epoch(self):
        self.model.eval()
        n_valid_batches = 20 

        if self.params.normalization == 'minmax':
            Exception("minmax normalization not supported")
            # mult = torch.as_tensor(np.load(self.params.global_stds_path)[0, self.params.out_channels, 0, 0]).to(self.device)
        elif self.params.normalization == 'zscore':
            mult = torch.as_tensor(np.load(self.params.global_stds_path)[0, self.params.out_channels, 0, 0]).to(self.device)
            # usag ???

        valid_buff = torch.zeros((3), dtype=torch.float32, device=self.device)
        valid_loss = valid_buff[0].view(-1) # 0
        valid_l1 = valid_buff[1].view(-1)   # 0
        valid_steps = valid_buff[2].view(-1) # 0

        valid_weighted_rmse = torch.zeros((self.params.N_out_channels), dtype=torch.float32, device=self.device)
        valid_weighted_acc = torch.zeros((self.params.N_out_channels), dtype=torch.float32, device=self.device)

        valid_start = time.time()
        sample_idx = np.random.randint(len(self.valid_data_loader))
        with torch.no_grad():
            for i, data in enumerate(self.valid_data_loader, 0):
                inp, tar = map(lambda x: x.to(self.device, dtype=torch.float), data)
                gen = self.model(inp)
                gen.to(self.device, dtype=torch.float)
                valid_loss += self.loss_obj(gen, tar)
                valid_l1 += nn.functional.l1_loss(gen, tar)

                valid_steps += 1.
                # save fields for vis before log norm
                if (i == sample_idx) and self.params.log_to_wandb:
                    fields = [gen[0, 0].detach().cpu().numpy(), tar[0, 0].detach().cpu().numpy()]

                os.makedirs(params['experiment_dir'] + "/" + str(i), exist_ok =True)
                save_image(torch.cat((gen[0, 0],
                                      torch.zeros((self.valid_dataset.img_shape_x, 4)).to(self.device, dtype=torch.float),
                                      tar[0, 0]), axis=1),
                           os.path.join(params['experiment_dir'], str(i), f's0_epoch_{self.epoch}.png'))

        if dist.is_initialized():
            dist.all_reduce(valid_buff)
            dist.all_reduce(valid_weighted_rmse)

        # divide by number of steps
        valid_buff[0:2] = valid_buff[0:2] / valid_buff[2]
        valid_weighted_rmse = valid_weighted_rmse / valid_buff[2]
        print('valid_weighted_rmse:', valid_weighted_rmse.shape)
        print('mult:', mult.shape)
        valid_weighted_rmse *= mult

        # download buffers
        valid_buff_cpu = valid_buff.detach().cpu().numpy()
        valid_weighted_rmse_cpu = valid_weighted_rmse.detach().cpu().numpy()

        valid_time = time.time() - valid_start
        valid_weighted_rmse = mult * torch.mean(valid_weighted_rmse, axis=0)
        
        logs = {'valid_loss': valid_buff_cpu[0],
                'valid_l1': valid_buff_cpu[1], 
                'valid_rmse_t0':  valid_weighted_rmse_cpu[0], 
                'valid_rmse_s0':  valid_weighted_rmse_cpu[6],
                'valid_rmse_u0':  valid_weighted_rmse_cpu[12],
                'valid_rmse_v0':  valid_weighted_rmse_cpu[18],
                'valid_rmse_ssh': valid_weighted_rmse_cpu[24]}
        if self.params.log_to_wandb:
            wandb.log(logs, step=self.epoch)

        return valid_time, logs

    def validate_final(self):
        self.model.eval()
        n_valid_batches = int(self.valid_dataset.n_patches_total / self.valid_dataset.n_patches)  # validate on whole dataset
        valid_weighted_rmse = torch.zeros(n_valid_batches, self.params.N_out_channels)

        if self.params.normalization == 'minmax':
            raise Exception("minmax normalization not supported")
        elif self.params.normalization == 'zscore':
            mult = torch.as_tensor(np.load(self.params.global_stds_path)[0, self.params.out_channels, 0, 0]).to(self.device)

        with torch.no_grad():
            for i, data in enumerate(self.valid_data_loader):
                if i > 100:
                    break
                inp, tar = map(lambda x: x.to(self.device, dtype=torch.float), data)

                gen = self.model(inp)
                gen.to(self.device, dtype=torch.float)

                valid_loss[i] += self.loss_obj(gen, tar)
                valid_l1[i] += nn.functional.l1_loss(gen, tar)

                for c in range(self.params.N_out_channels):
                    valid_weighted_rmse[i, c] = weighted_rmse_torch(gen[0, c], tar[0, c], self.device)

            # un-normalize
            valid_weighted_rmse = mult * torch.mean(valid_weighted_rmse[0:100], axis=0).to(self.device)

        return valid_weighted_rmse

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

        torch.save({'iters': self.iters, 'epoch': self.epoch, 'model_state': model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, checkpoint_path)

    def restore_checkpoint(self, checkpoint_path):
        """ We intentionally require a checkpoint_dir to be passed
            in order to allow Ray Tune to use this function """
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.params.local_rank))
        try:
            self.model.load_state_dict(checkpoint['model_state'])
        except:
            new_state_dict = OrderedDict()
            for key, val in checkpoint['model_state'].items():
                name = key[7:]
                new_state_dict[name] = val
            self.model.load_state_dict(new_state_dict)
        self.iters = checkpoint['iters']
        self.startEpoch = checkpoint['epoch']
        if self.params.resuming:  
        # restore checkpoint is used for finetuning as well as resuming. 
        # If finetuning (i.e., not resuming), restore checkpoint does not load optimizer state, instead uses config specified lr.
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default='00', type=str)
    parser.add_argument("--yaml_config", default='./config/AFNO.yaml', type=str)  # 配置文件路径
    parser.add_argument("--config", default='AFNO', type=str)
    parser.add_argument("--enable_amp", action='store_true')
    parser.add_argument("--epsilon_factor", default=0, type=float)
    parser.add_argument("--comm_rank", default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--local_rank', default=None, type = int,
                        help= 'OMPI_COMM_WORLD_LOCAL_RANK')

    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config, True)
    params['epsilon_factor'] = args.epsilon_factor

    params['world_size'] = args.world_size # 进程总数=节点数*每个节点的GPU数量

    # 配置每个worker的gpu
    if args.dist_url == "env://" and args.rank == -1:
        args.local_rank = int(os.environ["RANK"])
        local_rank = args.local_rank

    torch.backends.cudnn.benchmark = True
    world_rank = args.comm_rank
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method=args.dist_url,
                            world_size=args.world_size, rank=args.comm_rank)

    params['global_batch_size'] = params.batch_size
    params['batch_size'] = int(params.batch_size // params['world_size'])  # batch size 设置必须为单卡的 n 倍
    params['enable_amp'] = args.enable_amp  # 自动混合精度
    params['local_rank'] = args.local_rank  # GPU编号

    # Set up directory
    expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
    if world_rank == 0:
        os.makedirs(expDir, exist_ok =True)
        os.makedirs(os.path.join(expDir, 'training_checkpoints/'), exist_ok =True)

    params['experiment_dir'] = os.path.abspath(expDir)
    params['checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/ckpt.tar')  # 保存参数 loss epoch
    params['best_checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')

    # Do not comment this line out please:
    args.resuming = True if os.path.isfile(params.checkpoint_path) else False
    params['resuming'] = args.resuming

    # this will be the wandb name
    params['entity'] = "ocean_ai_model"  # team name 
    params['project'] = "ai4science"     # project name
    params['group'] = "025_daily"
    params['name'] = args.config + '_' + str(args.run_num)  # experiment name

    if world_rank == 0:
        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'out.log'))
        logging_utils.log_versions()
        params.log()

    params['log_to_wandb'] = (world_rank == 0) and params['log_to_wandb']
    params['log_to_screen'] = (world_rank == 0) and params['log_to_screen']

    params['in_channels'] = np.array(params['in_channels'])
    params['out_channels'] = np.array(params['out_channels'])
    params['N_out_channels'] = len(params['out_channels'])

    if world_rank == 0:
        hparams = ruamelDict()
        yaml = YAML()
        for key, value in params.params.items():
            hparams[str(key)] = str(value)
        with open(os.path.join(expDir, 'hyperparams.yaml'), 'w') as hpfile:
            yaml.dump(hparams, hpfile)

    trainer = Trainer(params, world_rank)
    trainer.train()
    logging.info('DONE ---- rank %d' % world_rank)
