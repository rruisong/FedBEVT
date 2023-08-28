import yaml
import os
import shutil
from tqdm import tqdm
import fnmatch
import random
import glob
import argparse


def fed_data_parser():
    """
    Parse the input arguments for federated data processing.
    Returns
    -------
    opt : dict
        Arguments for federated data processing.
    """
    parser = argparse.ArgumentParser(description="Distribute dataset in federated settings")
    parser.add_argument("--config", type=str, required=True, help='configuration file ')
    parser.add_argument("--source", type=str, required=True, help='source data directory')
    parser.add_argument("--target", type=str, required=True, help='target directory')
    opt = parser.parse_args()
    return opt


def main():
    """
    Main function for processing the train data in federated settings
    """
    opt = fed_data_parser()
    with open(os.path.join("./fedbevt/data_utils/config_fedsettings", opt.config), "r") as stream:
        try:
            config = yaml.safe_load(stream)

            # read the dir list in original data
            data_source = opt.source
            data_dir_list = os.listdir(data_source)
            data_dir_list.sort()

            # read the client configuration
            root_dir = opt.target
            if not os.path.exists(root_dir):
                os.mkdir(root_dir)

            root_dir_train = os.path.join(root_dir, 'train')
            if not os.path.exists(root_dir_train):
                os.mkdir(root_dir_train)

            root_dir_test = os.path.join(root_dir, 'test')
            if not os.path.exists(root_dir_test):
                os.mkdir(root_dir_test)

            for client_name in tqdm(config["clients"]["names"]):
                client_dir_train = os.path.join(root_dir, 'train', client_name)
                if not os.path.exists(client_dir_train):
                    os.mkdir(client_dir_train)

                client_dir_test = os.path.join(root_dir, 'test', client_name)
                if not os.path.exists(client_dir_test):
                    os.mkdir(client_dir_test)

                for train_data_folder in config["clients"][client_name]["train_dir_list"]:
                    destination_train = shutil.copytree(os.path.join(data_source, train_data_folder),
                                                        os.path.join(client_dir_train, train_data_folder))

                for test_data_folder in config["clients"][client_name]["test_dir_list"]:
                    if not config["clients"][client_name]["test_dir_list"]:
                        continue
                    destination_test = shutil.copytree(os.path.join(data_source, test_data_folder),
                                                        os.path.join(client_dir_test, test_data_folder))

        except yaml.YAMLError as exc:
            print(exc)


if __name__ == '__main__':
    main()
