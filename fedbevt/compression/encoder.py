import copy

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

__all__ = ['calc_msg_size', 'HuffmanEncoder', 'calc_entropy']


class NodeTree(object):
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def children(self):
        return self.left, self.right

    def __str__(self):
        return self.left, self.right


def huffman_code_tree(node, binString=''):
    """
    Function to find Huffman Code
    """
    if type(node) is str:
        return {node: binString}
    (l, r) = node.children()
    d = dict()
    d.update(huffman_code_tree(l, binString + '0'))
    d.update(huffman_code_tree(r, binString + '1'))
    return d


def make_tree(nodes):
    """
    Function to make tree
    :param nodes: Nodes
    :return: Root of the tree
    """
    while len(nodes) > 1:
        (key1, c1) = nodes[-1]
        (key2, c2) = nodes[-2]
        nodes = nodes[:-2]
        node = NodeTree(key1, key2)
        nodes.append((node, c1 + c2))
        nodes = sorted(nodes, key=lambda x: x[1], reverse=True)
    return nodes[0][0]


def close_to_any(a, floats, **kwargs):
  return np.any(np.isclose(a, floats, **kwargs))


class HuffmanEncoder(object):

    def __init__(self, delete_zero=1):
        self.delete_zero = delete_zero
        self.state_dict = None
        self.layer_name_list = [] # layer name list
        self.layer_dict = {} # layer name as key, parameters in the layer in one list as value
        self.model_param_list = []
        self._orig_bit = 0
        self._no_huffman_bit = 0
        self._entropy = 0

    def load_model_dict(self, model_state_dict):
        self.state_dict = copy.deepcopy(model_state_dict)
        self.layer_name_list = []  # layer name list
        self.layer_dict = {}  # layer name as key, parameters in the layer in one list as value
        self.model_param_list = []
        for layer_name in self.state_dict:
            self.layer_name_list.append(layer_name)
            self.layer_dict[layer_name] = torch.flatten(self.state_dict[layer_name]).tolist()
            self.model_param_list.extend(torch.flatten(self.state_dict[layer_name]).tolist())
        self._orig_bit = len(self.model_param_list) * 32

    def get_orig_bits(self):
        return self._orig_bit

    def get_non_huffman_bits(self):
        return self._no_huffman_bit

    def get_entropy(self):
        return self._entropy

    def _get_param_list(self):
        model_param_list = []
        for key in self.state_dict:
            model_param_list.extend(torch.flatten(self.state_dict[key]).tolist())
        return model_param_list

    def _get_onelayer_param_list(self, layer_id):
        key = self.key_name[0]
        model_param_list=torch.flatten(self.state_dict[key]).tolist()
        return model_param_list

    def isclose(self, a, b, rel_tol=1e-09, abs_tol=0.0):
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    def get_float_list(self, model_param_list):
        float_list = []
        float_freq = {}
        for ele in model_param_list:
            if close_to_any(ele, float_list, rtol=1e-03, atol=1e-03):
                float_freq[len(float_list) - 1] += 1
            else:
                float_list.append(ele)
                float_freq[len(float_list)-1] = 1
        # compute the probabilistic of each binary float64
        float_prob = {k: v / len(model_param_list) for k, v in float_freq.items()}
        float_prob = dict(sorted(float_prob.items(), key=lambda item: item[1]))

        return float_list, float_freq, float_prob

    def get_bfloat_list(self, model_param_list):
        bfloat_list = []
        bfloat_freq = {}
        for ele in model_param_list:
            # get the binary value of a float64

            ele_f32 = np.array(ele, dtype=np.float32).view(dtype=np.uint32)
            ele_b32 = bin(ele_f32)[2:]
            ele_b32 = '0' * (32 - len(ele_b32)) + ele_b32

            if self.delete_zero:
                if ele_b32 == '0'*32:
                    continue

            # get the freq list of the binary float64 and the corresponding list
            # The index of list is the key in the freq dict
            if ele_b32 not in bfloat_list:
                bfloat_list.append(ele_b32)
                bfloat_freq[len(bfloat_list)-1] = 1
            else:
                bfloat_freq[bfloat_list.index(ele_b32)] += 1

        # compute the probabilistic of each binary float64
        bfloat_prob = {k: v / len(model_param_list) for k, v in bfloat_freq.items()}
        bfloat_prob = dict(sorted(bfloat_prob.items(), key=lambda item: item[1]))

        return bfloat_list, bfloat_freq, bfloat_prob

    def get_huffman_bits(self):
        codebook_bits = 0
        weight_bits = 0
        index_bits = 0
        key_len = 0

        (bfloat_list, bfloat_freq, bfloat_prob) = self.get_bfloat_list(self.model_param_list)

        entopy = 0
        for key in bfloat_prob:
            p = bfloat_prob[key]
            entopy -= p * np.log2(p)
        self._entropy = entopy

        bfloat_freq_tuplist = []
        for key in bfloat_freq:
            bfloat_freq_tuplist.append((str(key), bfloat_freq[key]))
            # plt.plot(key, freq[key], "x")

        node = make_tree(bfloat_freq_tuplist)
        encoding = huffman_code_tree(node)

        bits = 0
        key_len += len(list(encoding.keys()))
        for i in encoding:
            # print(f'{i} : {encoding[i]} : {bfloat_freq[int(i)]}')
            # bits = sum of the frequency + sum of (frequency * index code) + sum of bits for individual key
            if self.delete_zero:
                weight_bits += bfloat_freq[int(i)] * 16 # use 16bits to save length of zero between two float
            weight_bits += len(encoding[i]) * bfloat_freq[int(i)]
            index_bits += bfloat_freq[int(i)]
        # code_book:
        codebook_bits += len(bfloat_list) * 32
        bits += index_bits + codebook_bits + weight_bits

        # without huffmancoding
        self._no_huffman_bit = len(bfloat_list) * 32 + np.log2(len(bfloat_list)) * len(bfloat_list) + np.log2(len(bfloat_list)) * self._orig_bit/32

        return key_len, bits, weight_bits, index_bits, codebook_bits


def calc_entropy(state_dict):
    def get_bfloat_list(model_param_list):
        bfloat_list = []
        bfloat_freq = {}
        for ele in model_param_list:
            # get the binary value of a float64

            ele_f32 = np.array(ele, dtype=np.float32).view(dtype=np.uint32)
            ele_b32 = bin(ele_f32)[2:]
            ele_b32 = '0' * (32 - len(ele_b32)) + ele_b32

            if ele_b32 == '0'*32:
                continue

            # get the freq list of the binary float64 and the corresponding list
            # The index of list is the key in the freq dict
            if ele_b32 not in bfloat_list:
                bfloat_list.append(ele_b32)
                bfloat_freq[len(bfloat_list)-1] = 1
            else:
                bfloat_freq[bfloat_list.index(ele_b32)] += 1

        # compute the probabilistic of each binary float64
        bfloat_prob = {k: v / len(model_param_list) for k, v in bfloat_freq.items()}
        bfloat_prob = dict(sorted(bfloat_prob.items(), key=lambda item: item[1]))

        return bfloat_list, bfloat_freq, bfloat_prob

    model_param_list = []
    for key in state_dict:
        model_param_list.extend(torch.flatten(state_dict[key]).tolist())

    (bfloat_list, bfloat_freq, bfloat_prob) = get_bfloat_list(model_param_list)

    entopy = 0
    for key in bfloat_prob:
        p = bfloat_prob[key]
        entopy -= p * np.log2(p)
    return entopy


def calc_msg_size(state_dict):
    if state_dict is None:
        return 0

    he = HuffmanEncoder()
    he.load_model_dict(state_dict)
    key_len, bits, weight_bits, index_bits, codebook_bits = he.get_huffman_bits()
    return he.get_orig_bits() / bits


