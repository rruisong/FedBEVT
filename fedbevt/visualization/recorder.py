import numpy as np
import matplotlib.pyplot as plt
import json
from json import JSONEncoder
import pickle

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


class Recorder(object):
    def __init__(self, client_list=[]):
        self.res_list = []
        self.res = {'server': {'iid_accuracy': [], 'train_loss': [], 'avg_loss': [], 'valid_ave_loss': [],
                               'dynamic_ave_iou': []},
                    'clients': {}}
        for client_id in client_list:
            print(client_id)
            self.res['clients'][client_id] = {'iid_accuracy': [],
                                              'global_round': [],
                                              'train_loss': [],
                                              'avg_loss': [],
                                              'valid_ave_loss': [],
                                              'dynamic_ave_iou': []}

    def load(self, filename, label):
        with open(filename) as json_file:
            res = json.load(json_file, object_hook=as_python_object)
        self.res_list.append((res, label))

    def plot(self):
        fig, axes = plt.subplots(3, 4)
        for i, (res, label) in enumerate(self.res_list):
            for j, client in enumerate(res["clients"]):
                axes[0, j].plot(np.array(res['clients'][client]['global_round']),
                                np.array(res['clients'][client]['train_loss']), '-', label=label, alpha=1, linewidth=1)
                axes[1, j].plot(np.array(res['clients'][client]['global_round']),
                                np.array(res['clients'][client]['valid_ave_loss']), '-', label=label, alpha=1,
                                linewidth=1)
                axes[2, j].plot(np.array(res['clients'][client]['global_round']),
                                np.array(res['clients'][client]['dynamic_ave_iou']), '-', label=label, alpha=1,
                                linewidth=1)

        for i, ax_i in enumerate(axes):
            for j, ax in enumerate(ax_i):
                ax.set_xlabel('# of communication rounds', size=fontsize)
                if i == 0:
                    ax.set_ylabel('avg_loss', size=fontsize)
                if i == 1:
                    ax.set_ylabel('valid_ave_loss', size=fontsize)
                if i == 2:
                    ax.set_ylabel('dynamic_ave_iou', size=fontsize)
                ax.legend(prop={'size': fontsize})
                ax.tick_params(axis='both', labelsize=fontsize)
                ax.grid()
