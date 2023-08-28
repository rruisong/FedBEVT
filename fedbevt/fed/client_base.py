import argparse
import os
import statistics

import torch
import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import numpy as np

import fedbevt.config.yaml_utils as yaml_utils
from fedbevt.tools import train_utils
from fedbevt.tools import multi_gpu_utils
from fedbevt.data_utils.datasets import build_dataset
from fedbevt.utils.seg_utils import cal_iou_training

import matplotlib.pyplot as plt
import copy




class FedClient(object):
    def __init__(self, name, config, opt):
        """
        Initialize a client for federated learning.
        Parameters
        ----------
        name : str
            name of the current client.
        config: dict
            configuration of fedbevt
        opt: dict
            Arguments of fedbevt
        """
        self.config = copy.deepcopy(config)
        self.opt = copy.deepcopy(opt)

        self.target_ip = '127.0.0.3'
        self.port = 9999
        self.name = name

        self._batch_size = 64
        self._lr = 0.001
        self._momentum = 0.9
        self.num_workers = 2
        self.trainset = None
        self.train_loader = None
        self.testset = None
        self.test_data = None

        self.model = train_utils.create_model(self.config)
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.param_len = sum([np.prod(p.size()) for p in model_parameters])

        # optimizer setup
        self.optimizer = None

        # lr scheduler setup
        self.epoches = None
        self.num_steps = None
        self.scheduler = None

        self.loss_rec = []
        self.n_data = 0
        self.rsu_id = None
        gpu = opt.gpu
        self._device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

    def load_trainset(self, trainset):
        """
        Load the train dataset.
        Parameters
        ----------
        trainset: Dataset
            Train dataset for current client.
        """
        self.trainset = trainset
        self.n_data = len(trainset)

        self.train_loader = DataLoader(self.trainset,
                                       batch_size=self.config['train_params'][
                                           'batch_size'],
                                       num_workers=8,
                                       collate_fn=self.trainset.collate_batch,
                                       shuffle=True,
                                       pin_memory=False,
                                       drop_last=True)

        # optimizer setup
        self.optimizer = train_utils.setup_optimizer(self.config, self.model)

        # lr scheduler setup
        self.epoches = self.config['train_params']['epoches']
        self.num_steps = len(self.train_loader)
        self.scheduler = train_utils.setup_lr_schedular(self.config, self.optimizer, self.num_steps)

    def load_testset(self, testset):
        """
        Load the test dataset.
        Parameters
        ----------
        testset: Dataset object
            Test dataset for current client.
        """
        self.testset = testset

    def update(self, model_state_dict):
        """
        Update the local model w.r.t. predefined personalization rules.
        Parameters
        ----------
        model_state_dict: dict
            global model state dict from server
        """
        if self.opt.per == "cap":
            global_model_state_dict = self.cap_head_update(model_state_dict)
        elif self.opt.per == 'avg':
            global_model_state_dict = copy.deepcopy(model_state_dict)
        else:
            print("unexpected personlization")
            exit(1)
        self.model.load_state_dict(global_model_state_dict)

    def train(self, global_round):
        """
        Train model on local dataset.
        Parameters
        ----------
        global_round : int
            current global round number.

        Returns
        -------
        self.model.state_dict() : dict
            Local model state dict.
        self.n_data : int
            Number of local data points.
        final_loss.data.cpu().numpy() : float
            Train loss value.

        """
        self.model.to(self._device)
            
        model_without_ddp = self.model

        saved_path = self.opt.model_dir

        # record training
        writer = SummaryWriter(saved_path)

        # define the loss
        criterion = train_utils.create_loss(self.config)

        # half precision training
        if self.opt.half:
            scaler = torch.cuda.amp.GradScaler()

        # used to help schedule learning rate
        for epoch in range(1, self.epoches + 1):

            pbar2 = tqdm.tqdm(total=len(self.train_loader), leave=True)

            for i, batch_data in enumerate(self.train_loader):
                self.model.train()
                self.model.zero_grad()
                self.optimizer.zero_grad()

                batch_data = train_utils.to_device(batch_data, self._device)

                if not self.opt.half:
                    ouput_dict = self.model(batch_data['ego'])
                    final_loss = criterion(ouput_dict,
                                           batch_data['ego'])
                else:
                    with torch.cuda.amp.autocast():
                        ouput_dict = self.model(batch_data['ego'])
                        final_loss = criterion(ouput_dict, batch_data['ego'])

                criterion.logging(global_round, self.name, epoch, i, len(self.train_loader), writer,
                                  pbar=pbar2)
                
                # update tensorboard
                for lr_idx, param_group in enumerate(self.optimizer.param_groups):
                    writer.add_scalar('lr_%d' % lr_idx, param_group["lr"],
                                      epoch * self.num_steps + i)

                if not self.opt.half:
                    final_loss.backward()
                    self.optimizer.step()
                else:
                    scaler.scale(final_loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                    
                self.scheduler.step_update(epoch * self.num_steps * global_round + i)

            self.trainset.reinitialize()
            pbar2.update(1)
        self.model.to('cpu')

        return self.model.state_dict(), self.n_data, final_loss.data.cpu().numpy()
    
    def train_sim(self, global_round):
        """
        Train model on local dataset with SIM.
        Parameters
        ----------
        global_round : int
            current global round number.

        Returns
        -------
        self.model.state_dict() : dict
            Local model state dict.
        self.n_data : int
            Number of local data points.
        final_loss.data.cpu().numpy() : float
            Train loss value.

        """
        self.model.to(self._device)
            
        model_without_ddp = self.model

        saved_path = self.opt.model_dir

        # record training
        writer = SummaryWriter(saved_path)

        # define the loss
        criterion = train_utils.create_loss(self.config)

        # half precision training
        if self.opt.half:
            scaler = torch.cuda.amp.GradScaler()

        # used to help schedule learning rate
        for epoch in range(1, self.epoches + 1):

            pbar2 = tqdm.tqdm(total=len(self.train_loader), leave=True)
            
            for i, batch_data in enumerate(self.train_loader):
                tmp_model = copy.deepcopy(self.model)
                for stage in [1,2]:

                    self.model.train()
                    self.model.zero_grad()
                    self.optimizer.zero_grad()

                    batch_data = train_utils.to_device(batch_data, self._device)

                    if not self.opt.half:
                        ouput_dict = self.model(batch_data['ego'])
                        final_loss = criterion(ouput_dict,
                                            batch_data['ego'])
                    else:
                        with torch.cuda.amp.autocast():
                            ouput_dict = self.model(batch_data['ego'])
                            final_loss = criterion(ouput_dict, batch_data['ego'])
                    
                    if stage == 2:
                        # update tensorboard
                        for lr_idx, param_group in enumerate(self.optimizer.param_groups):
                            writer.add_scalar('lr_%d' % lr_idx, param_group["lr"],
                                            epoch * self.num_steps + i)

                    if not self.opt.half:
                        final_loss.backward()
                        self.optimizer.step()
                    else:
                        scaler.scale(final_loss).backward()
                        scaler.step(self.optimizer)
                        scaler.update()
                    
                    if stage == 1:
                        self.model.load_state_dict(self.cap_head_update(tmp_model.state_dict()))
                    else: # staget == 2
                        self.model.load_state_dict(self.cap_tail_update(tmp_model.state_dict()))
                
                
                criterion.logging(global_round, self.name, epoch, i, len(self.train_loader), writer,
                                    pbar=pbar2)
                self.scheduler.step_update(epoch * self.num_steps * global_round + i)

                pbar2.update(1)
            self.trainset.reinitialize()
        self.model.to('cpu')

        return self.model.state_dict(), self.n_data, final_loss.data.cpu().numpy()

    def save_model(self, global_round, res_name):
        """
        Save local model
        Parameters
        ----------
        global_round : int
            current global round number.
        res_name : str
            result name as prefix for directory of saved models
            
        """
        saved_path = self.opt.model_dir
        saved_path = os.path.join(saved_path, "model_results", res_name, self.name)
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)
        if global_round % self.config['train_params']['save_freq'] == 0:
            torch.save(self.model.state_dict(),
                       os.path.join(saved_path,
                                    'net_global_round%d.pth' % (global_round)))

    def cap_head_update(self, model_state_dict):
        """
        CaP head update.
        Parameters
        ----------
        model_state_dict : dict
            global model state dict in current global round.

        Returns
        -------
        global_model_state_dict : dict
            rebuild global model for CaP local model update.
        """
        global_model_state_dict = copy.deepcopy(model_state_dict)
        local_model_state_dict = copy.deepcopy(self.model.state_dict())
        for key in global_model_state_dict:
            if key[:27] == 'fax.cross_views.0.cam_embed' \
                    or key[:27] == 'fax.cross_views.1.cam_embed' \
                    or key[:27] == 'fax.cross_views.2.cam_embed' \
                    or key[:27] == 'fax.cross_views.0.img_embed' \
                    or key[:27] == 'fax.cross_views.1.img_embed' \
                    or key[:27] == 'fax.cross_views.2.img_embed':
                global_model_state_dict[key] = local_model_state_dict[key]

        return global_model_state_dict
    
    def cap_tail_update(self, model_state_dict):
        """
        CaP tail update.
        Parameters
        ----------
        model_state_dict : dict
            global model state dict in current global round.

        Returns
        -------
        global_model_state_dict : dict
            rebuild global model for non-CaP local model update.
        """
        global_model_state_dict = copy.deepcopy(model_state_dict)
        local_model_state_dict = copy.deepcopy(self.model.state_dict())
        for key in global_model_state_dict:
            if key[:27] == 'fax.cross_views.0.cam_embed' \
                    or key[:27] == 'fax.cross_views.1.cam_embed' \
                    or key[:27] == 'fax.cross_views.2.cam_embed' \
                    or key[:27] == 'fax.cross_views.0.img_embed' \
                    or key[:27] == 'fax.cross_views.1.img_embed' \
                    or key[:27] == 'fax.cross_views.2.img_embed':
                pass
            else:
                global_model_state_dict[key] = local_model_state_dict[key]

        return global_model_state_dict

    def local_test(self, global_round):
        """
        Test on local dataset.
        Parameters
        ----------
        global_round : int
            current global round number.

        Returns
        -------
        test_ave_loss: float
            test average loss value.
        dynamic_ave_iou: float
            average intersection over union for vehicle objects.

        """
        # define the loss
        criterion = train_utils.create_loss(self.config)

        test_loader = DataLoader(self.testset,
                                 batch_size=self.config['train_params']['batch_size'],
                                 num_workers=8,
                                 collate_fn=self.testset.collate_batch,
                                 shuffle=False,
                                 pin_memory=False,
                                 drop_last=True)

        self.model.to(self._device)

        test_ave_loss = []
        dynamic_ave_iou = []

        with torch.no_grad():
            for i, batch_data in enumerate(test_loader):
                self.model.eval()

                batch_data = train_utils.to_device(batch_data, self._device)
                output_dict = self.model(batch_data['ego'])

                final_loss = criterion(output_dict, batch_data['ego'])
                test_ave_loss.append(final_loss.item())

                # visualization purpose
                output_dict = self.testset.post_process(batch_data['ego'], output_dict)
                iou_dynamic, iou_static = cal_iou_training(batch_data,
                                                           output_dict)
                dynamic_ave_iou.append(iou_dynamic[1])

        test_ave_loss = statistics.mean(test_ave_loss)
        dynamic_ave_iou = statistics.mean(dynamic_ave_iou)

        print('At global_round %d, the test loss is %f,'
              'the dynamic iou is %f'
              % (global_round,
                 test_ave_loss,
                 dynamic_ave_iou))

        return test_ave_loss, dynamic_ave_iou
