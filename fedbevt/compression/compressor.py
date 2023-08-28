import torch

from models import *
from copy import deepcopy

import numpy as np
import random

__all__ = ['sparsify', 'quantize']


def sparsify(state_dict, spar_ratio):
    model_param_list = []
    for layer_name in state_dict:
        model_param_list.extend(torch.flatten(state_dict[layer_name]).tolist())
    tensorlen = len(model_param_list)
    torch_param = torch.FloatTensor(model_param_list)
    topk = torch.topk(torch.abs(torch_param), max(int(tensorlen * spar_ratio), 1), 0, largest=True, sorted=False)
    threshold = torch.min(topk[0])
    for key in state_dict:
        state_dict[key][torch.abs(state_dict[key]) < threshold] = 0
        # state_dict[key] = state_dict[key].type(torch.float16)

    return state_dict


def quantize(state_dict, qe):
    model_param_list = []
    for layer_name in state_dict:
        model_param_list.extend(torch.flatten(state_dict[layer_name]).tolist())
    torch_param = torch.FloatTensor(model_param_list)
    i = 0
    md = torch.median(torch.abs(torch_param[torch_param != 0]))
    threshold = md
    torch_param[torch.abs(torch_param) < threshold] = 0
    n = torch.sum(torch.abs(torch_param) > threshold)
    torch_param[torch_param > 0] = encode_median(torch_param[torch_param > 0], i, qe)
    torch_param[torch_param < 0] = encode_median(torch_param[torch_param < 0], i, qe)

    start_ind = 0
    end_ind = 0
    for layer_name in state_dict:
        end_ind += state_dict[layer_name].nelement()
        state_dict[layer_name] = torch.FloatTensor(torch_param[start_ind:end_ind]).reshape(
            state_dict[layer_name].size())
        start_ind = end_ind

    return state_dict


def encode_median(state_dict, i, qe):
    i = 1 + i
    if state_dict.nelement() == 0:
        return state_dict
    median = torch.median(state_dict)
    if i < qe:
        state_dict[state_dict > median] = encode_median(state_dict[state_dict > median], i, qe)
        state_dict[state_dict <= median] = encode_median(state_dict[state_dict <= median], i, qe)
    else:
        state_dict = median
    return state_dict
