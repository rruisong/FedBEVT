import numpy as np
import argparse
import os
import statistics
import copy

import torch
import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler

import fedbevt.config.yaml_utils as yaml_utils
from fedbevt.tools import train_utils
from fedbevt.tools import multi_gpu_utils
from fedbevt.data_utils.datasets import build_dataset
from fedbevt.utils.seg_utils import cal_iou_training


class FedServer(object):
    def __init__(self, client_list, config, opt):
        """
        Initialize a server for federated learning.
        Parameters
        ----------
        client_list: list
            A list of clients for federated learning.
        config: dict
            configuration of fedbevt
        opt: dict
            Arguments of fedbevt
        """
        self.config = config
        self.opt = opt
        self.client_state = {}
        self.client_loss = {}
        self.client_n_data = {}
        self.selected_clients = []
        self._batch_size = 200

        self.client_list = client_list
        self.testset = None

        self.round = 0
        self.n_data = 0
        gpu = gpu = opt.gpu
        self._device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

        self.model = train_utils.create_model(self.config)

    def load_testset(self, testset):
        """
        Load the test dataset.
        Parameters
        ----------
        testset: Dataset object
            Test dataset for current client.
        """
        self.testset = testset
        self.n_data = len(testset)

    def state_dict(self):
        """

        Returns
        -------

        """
        return self.model.state_dict()

    def test(self, global_round, avg_loss):
        """
        Test the global model.
        Parameters
        ----------
        global_round : int
            current global round number.
        ave_loss: float
            average loss values from clients.

        Returns
        -------
        avg_loss : float
            average train loss across clients.
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
        accuracy_collector = 0

        test_ave_loss = []
        dynamic_ave_iou = []
        static_ave_iou = []
        lane_ave_iou = []

        with torch.no_grad():
            for i, batch_data in enumerate(test_loader):
                self.model.eval()

                batch_data = train_utils.to_device(batch_data, self._device)
                output_dict = self.model(batch_data['ego'])

                final_loss = criterion(output_dict,
                                       batch_data['ego'])
                test_ave_loss.append(final_loss.item())

                # visualization purpose
                output_dict = \
                    self.testset.post_process(batch_data['ego'],
                                             output_dict)
                # train_utils.save_bev_seg_binary(output_dict, batch_data, saved_path, i, global_round)
                iou_dynamic, iou_static = cal_iou_training(batch_data,
                                                           output_dict)
                static_ave_iou.append(iou_static[1])
                dynamic_ave_iou.append(iou_dynamic[1])
                lane_ave_iou.append(iou_static[2])

        test_ave_loss = statistics.mean(test_ave_loss)
        dynamic_ave_iou = statistics.mean(dynamic_ave_iou)

        print('At global_round %d, the test loss is %f, the dynamic iou is %f' % 
                                        (global_round,
                                         test_ave_loss,
                                         dynamic_ave_iou))

        return avg_loss, test_ave_loss, dynamic_ave_iou
    
    def agg(self):
        """
        Aggregate the models from clients
        Returns
        -------
        model_state : dict
            global aggregated model state
        avg_loss : float
            average train loss for all clients
        self.n_data : int
            Number of total data points
        """
        client_num = len(self.client_list)

        if client_num == 0 or self.n_data == 0:
            return self.model.state_dict(), 0, 0

        model_state = self.model.state_dict()
        avg_loss = 0
        # print('number of selected clients in Cloud: ' + str(client_num))

        for i, name in enumerate(self.client_list):
            if name not in self.client_state:
                continue
            for key in self.client_state[name]:
                if i == 0:
                    model_state[key] = self.client_state[name][key] * self.client_n_data[name] / self.n_data
                else:
                    model_state[key] = model_state[key] + self.client_state[name][key] * self.client_n_data[
                        name] / self.n_data

            avg_loss = avg_loss + self.client_loss[name] * self.client_n_data[name] / self.n_data

        self.model.load_state_dict(model_state)
        self.round = self.round + 1

        return model_state, avg_loss, self.n_data

    def save_model(self, global_round, res_name):
        """
        save global models
        Parameters
        ----------
        global_round : int
            current global round number.
        res_name : str
            result name as prefix for directory of saved models
        """
        saved_path = self.opt.model_dir
        saved_path = os.path.join(saved_path, "model_results", res_name, 'global')
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)
        if global_round % self.config['train_params']['save_freq'] == 0:
            torch.save(self.model.state_dict(),
                       os.path.join(saved_path,
                                    'net_global_round%d.pth' % (global_round + 1)))

    def rec(self, name, state_dict, n_data, loss):
        """
        Receive the local models from connected clients.
        Parameters
        ----------
        name : str
            client name.
        state_dict : dict
            uploaded local model from a dedicated client.
        n_data : int
            number of data points in a dedicated client.
        loss : float
            train loss of a dedicated client.
        """
        self.n_data = self.n_data + n_data
        self.client_state[name] = {}
        self.client_n_data[name] = {}

        self.client_state[name].update(state_dict)
        self.client_n_data[name] = n_data
        self.client_loss[name] = {}
        self.client_loss[name] = loss

    def flush(self):
        """
        Flush the information for current communication round.
        """
        self.n_data = 0
        self.client_state = {}
        self.client_n_data = {}
        self.client_loss = {}
