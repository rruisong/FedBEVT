import argparse
import os
import random
import statistics

import torch

import fedbevt.config.yaml_utils as yaml_utils
from fedbevt.tools import train_utils
from fedbevt.tools import multi_gpu_utils
from fedbevt.data_utils.datasets import build_dataset

from fedbevt.fed.client_base import FedClient
from fedbevt.fed.server_base import FedServer

from datetime import datetime
from datetime import date

import json
from json import JSONEncoder
import pickle
from tqdm import tqdm
import numpy as np

from fedbevt.visualization.recorder import Recorder

SIM = 0

json_types = (list, dict, str, int, float, bool, type(None))


class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, json_types):
            return super().default(self, obj)
        return {'_python_object': pickle.dumps(obj).decode('latin-1')}


def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(dct['_python_object'].encode('latin-1'))
    return dct


def train_parser():
    """
    Parse the input arguments for FedBEVT
    Returns
    -------
    opt : dict
        Arguments for FedBEVT
    """
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--config", type=str, required=True,
                        help='configuration file ')
    parser.add_argument('--gpu', default=0, type=int,
                    help='GPU used for training')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument("--half",  action='store_true',
                        help="whether train with half precision")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for training')
    parser.add_argument('--per', default='avg', type=str,
                        help='method for personalization')
    opt = parser.parse_args()

    return opt


def main():
    """
    Main function for FedBEVT
    """
    opt = train_parser()

    today = date.today()
    now = datetime.now().time()
    current_time = today.strftime("%Y%m%d_") + now.strftime("%H%M%S")
    res_dir = os.path.join('results', 'json_logs')
    res_file_name = 'FedBEVT_' + '_' + str(opt.gpu) + '_' + opt.per + '_' + current_time
    res_file_dir = os.path.join(res_dir, res_file_name)

    config = yaml_utils.load_yaml(os.path.join("./fedbevt/config/",opt.config), opt)

    seed = train_utils.init_random_seed(None if opt.seed == 0 else opt.seed)
    config['train_params']['seed'] = seed
    print('Set seed to %d' % seed)
    train_utils.set_random_seed(seed)
    torch.cuda.set_device(opt.gpu)

    # fedbevt_test_dataset_server = build_dataset(config, visualize=False, train=True, test=True)

    root_dir = config['root_dir']
    test_dir = config['test_dir']

    client_dict = {}
    client_list = config['clients']['client_list']

    # Initialize the clients
    for client in client_list:
        if not os.path.exists(os.path.join(root_dir, client)):
            print("client: %s defined in yaml file not found in the path %s" % client, root_dir)
        if not os.path.exists(os.path.join(test_dir, client)):
            print("client: %s defined in yaml file not found in the path %s" % client, test_dir)

    for client in client_list:
        config['root_dir'] = os.path.join(root_dir, client)
        config['test_dir'] = os.path.join(test_dir, client)
        fedbevt_train_dataset = build_dataset(config, visualize=False, train=True, client=client)

        fed_client = FedClient(client, config, opt)
        fed_client.load_trainset(fedbevt_train_dataset)
        fedbevt_test_dataset_client = build_dataset(config, visualize=False, train=True, test=True, client=client)
        fed_client.load_testset(fedbevt_test_dataset_client)

        client_dict[client] = fed_client
    recorder = Recorder(client_list)
    print("Detect the clients:" + str(client_list))

    # Initialize the server
    fed_server = FedServer(client_list, config, opt)
    # fed_server.load_testset(fedbevt_test_dataset_server)
    global_state_dict = fed_server.state_dict()

    # Start training
    for global_round in range(1, config['train_params']['num_rounds']+1):
        selected_clients = np.random.choice(client_list, size=int((1 - config['network']['straggler_ratio']) * len(client_list)), replace=False) 
        for client in client_list:
            fed_server.save_model(global_round, res_name=res_file_name)
            client_dict[client].update(global_state_dict)
            if global_round % config['train_params']['eval_freq'] == 0:
                valid_ave_loss, dynamic_ave_iou = client_dict[client].local_test(global_round)
                recorder.res['clients'][client]['global_round'].append(global_round)
                recorder.res['clients'][client]['valid_ave_loss'].append(valid_ave_loss)
                recorder.res['clients'][client]['dynamic_ave_iou'].append(dynamic_ave_iou)
            if client in selected_clients:
                if SIM:
                    state_dict, n_data, loss = client_dict[client].train_sim(global_round)
                else:
                    state_dict, n_data, loss = client_dict[client].train(global_round)
                fed_server.rec(client_dict[client].name, state_dict, n_data, loss)
            client_dict[client].save_model(global_round, res_file_name)
        global_state_dict, avg_loss, _ = fed_server.agg()

        # avg_loss, valid_ave_loss, dynamic_ave_iou = fed_server.val(global_round, avg_loss)

        fed_server.flush()
        if global_round % config['train_params']['eval_freq'] == 0:
            recorder.res['server']['avg_loss'].append(avg_loss)
        # recorder.res['server']['valid_ave_loss'].append(valid_ave_loss)
        # recorder.res['server']['dynamic_ave_iou'].append(dynamic_ave_iou)

        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        if global_round % 1 == 0:
            print('FedBEVT_'+ current_time)
            with open(res_file_dir,"w") as jsfile:
                json.dump(recorder.res, jsfile, cls=PythonObjectEncoder)

    with open(res_file_dir, "w") as jsfile:
        json.dump(recorder.res, jsfile, cls=PythonObjectEncoder)


if __name__ == '__main__':
    main()
