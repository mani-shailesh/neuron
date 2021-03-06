import cPickle
import os
import random

import util

import numpy as np

DATA_DIR = "../data"
NUM_TRAIN_FILES = 5


class DataStore:
    """
    Class to handle all raw data related operations
    """
    def __init__(self, data_dir=DATA_DIR, num_train_files=NUM_TRAIN_FILES, val_fraction=0.2):
        self.data_dir = data_dir
        self.num_train_files = num_train_files
        """
        Number of files for training data
        """
        self.data_dict = None
        """
        A dictionary with 'train', 'val' & 'test' as keys
        Each value in this dictionary in turn is a dictionary with following structure
        'data': numpy array of data
        'labels':   numpy array of labels in same order as data
        """
        self.label_details_dict = None
        """
        Dictionary with label indices as keys.
        Each value is a dictionary with following keys:
        'name'
        """
        self.validation_fraction = val_fraction
        """
        Fraction of training data to be split and used as validation set
        """
        self.zero_centered = False
        self.normalized = False

    def load_data(self, split_val=True):
        """
        Loads training and test data from the `data_dir` and resets 'data_dict' & 'label_details_dict'
        :param split_val: Split training into training and Validation if True
        """
        print("Loading data...")

        label_data_dict = {}
        self.label_details_dict = {}
        meta_filename = "batches.meta"
        label_name_list = unpickle(os.path.join(self.data_dir, meta_filename))['label_names']
        for ii, label_name in enumerate(label_name_list):
            label_data_dict[ii] = {
                'train_instance_list':  [],
                'val_instance_list': [],
                'test_instance_list':   []
            }
            self.label_details_dict[ii] = {
                'name': label_name
            }

        base_filename = "data_batch_"
        for ii in range(1, self.num_train_files+1):
            filename = base_filename + str(ii)
            train_data_dict = unpickle(os.path.join(self.data_dir, filename))
            for jj, label_idx in enumerate(train_data_dict['labels']):
                label_data_dict[label_idx]['train_instance_list'].append(train_data_dict['data'][jj])

        test_filename = "test_batch"
        test_data_dict = unpickle(os.path.join(self.data_dir, test_filename))
        for jj, label_idx in enumerate(test_data_dict['labels']):
            label_data_dict[label_idx]['test_instance_list'].append(test_data_dict['data'][jj])

        print("Done.")

        if split_val:
            label_data_dict = self.create_validation_set(label_data_dict)
            key_list = ['train', 'val', 'test']
        else:
            key_list = ['train', 'test']

        self.data_dict = {
            key: {'data': [], 'labels': []} for key in key_list
            }

        print("Converting your data into consumable format...")
        # populate data_dict
        for label_idx in label_data_dict.keys():
            for key in key_list:
                self.data_dict[key]['data'].extend(label_data_dict[label_idx][key + '_instance_list'])
                self.data_dict[key]['labels'].extend(
                    [label_idx for _ in range(len(label_data_dict[label_idx][key + '_instance_list']))]
                )

        for key in key_list:
            self.data_dict[key]['data'] = np.array(self.data_dict[key]['data'], dtype=np.float)
            self.data_dict[key]['labels'] = np.array(self.data_dict[key]['labels'])
        print("Done")

        self.zero_centered = False
        self.normalized = False

    def create_validation_set(self, label_data_dict):
        """
        Splits the training data loaded in 'label_data_dict' into training and validation
        Updates 'label_data_dict' accordingly
        """
        print("Splitting training set into training and validation sets...")
        for label in label_data_dict.keys():
            train_instance_list = label_data_dict[label]['train_instance_list']
            random.shuffle(train_instance_list)
            num_validation = int(len(train_instance_list) * self.validation_fraction)
            label_data_dict[label]['val_instance_list'] = train_instance_list[0:num_validation]
            label_data_dict[label]['train_instance_list'] = train_instance_list[num_validation:]
        print("Done.")
        return label_data_dict

    def get_data(self, zero_centre=True, normalize=True, split_val=True):
        """
        Return training, validation and test data with or without pre-processing
        :param zero_centre: Perform zero centering on data if this is True
        :param normalize: Normalize the data if this is True
        :param split_val: Split training into training and Validation if True
        :return: A dictionary with 'train', 'val' & 'test' as keys
        Each value in this dictionary in turn is a dictionary with following structure
        'data': numpy array of data
        'labels':   numpy array of labels in same order as data
        """

        # Make sure that data is loaded
        if self.data_dict is None:
            self.load_data(split_val)
        else:
            if split_val != ('val' in self.data_dict) or (self.zero_centered and not zero_centre) \
              or (self.normalized and not normalize):
                self.load_data(split_val)

        if split_val:
            key_list = ['train', 'val', 'test']
        else:
            key_list = ['train', 'test']

        if zero_centre:
            print("Zero centering the data...")
            train_mean = np.mean(self.data_dict['train']['data'], axis=0)
            for key in key_list:
                if len(self.data_dict[key]['data']) != 0:
                    self.data_dict[key]['data'] -= train_mean
            self.zero_centered = True
            print("Done.")

        if normalize:
            print("Normalizing the data...")
            train_std_dev = np.std(self.data_dict['train']['data'], axis=0)
            for key in key_list:
                self.data_dict[key]['data'] /= train_std_dev
            self.normalized = True
            print("Done.")

        return self.data_dict


class RNNDataStore:
    """
    DataStore class to load data for training of RNNs
    """
    def __init__(self, data_dir=DATA_DIR):
        """
        Initializer for the RNNDataStore class
        :param data_dir: Directory to read data files from
        """
        self.data_dir = data_dir
        self.train_set_list = []
        self.test_set_list = []

        self.train_mean = None
        self.std_dev = None
        self.squashed = False

    def load_data(self):
        """
        Load data from files into the memory
        """
        train_set_list, test_set_list = [], []

        with open(os.path.join(self.data_dir, 'train.txt')) as train_file:
            train_set = []
            last_num = 0
            for idx, line in enumerate(train_file.readlines()):
                num, val = line.strip().split(',')
                num, val = int(num), float(val)
                if idx > 0 and num != last_num + 1 and len(train_set) > 0:
                    train_set_list.append(train_set)
                    train_set = []
                train_set.append((num, val))
                last_num = num
            train_set_list.append(train_set)

        with open(os.path.join(self.data_dir, 'test.txt')) as test_file:
            test_set = []
            last_num = 0
            for idx, line in enumerate(test_file.readlines()):
                num, val = line.strip().split(',')
                num, val = int(num), float(val)
                if idx > 0 and num != last_num + 1 and len(test_set) > 0:
                    test_set_list.append(test_set)
                    test_set = []
                test_set.append((num, val))
                last_num = num
            test_set_list.append(test_set)

        self.train_set_list = np.array(train_set_list)
        self.test_set_list = np.array(test_set_list)

    def get_data(self, sequence_len, input_dim, output_dim_train, output_dim_val, zero_centre=True, normalize=True, val_fraction=0.3, min_zero_max_one=False):
        """
        Return loaded data in proper format
        :param sequence_len: Number of time steps in each instance
        :param input_dim: Dimension of input at each time step
        :param output_dim_train: Dimension of output for each instance in training set
        :param output_dim_val: Dimension of output for each instance in validation set
        :param zero_centre: Perform zero centering on data if this is True
        :param normalize: Normalize the data if this is True
        :param val_fraction: Fraction of training data to be returned as validation data
        :param min_zero_max_one: Squash the data between 0 and 1 if this is True
        :return: tuple : ((train_x, train_y), (val_x), (val_y))
        where train_x, val_x are numpy arrays of shape (None, sequence_len, input_dim),
        train_y, val_y are numpy array of shape (None, output_dim)
        """
        train_x = None
        train_y = None
        val_x = None
        val_y = None

        complete_train_set = []

        for train_set in self.train_set_list:
            split_idx = int(len(train_set) * (1 - val_fraction))

            complete_train_set.extend(train_set[0:split_idx, 1])

            formatted_train_set = util.convert_to_time_series(
                train_set[0:split_idx, 1], sequence_len, input_dim, output_dim_train
            )

            formatted_val_set = util.convert_to_time_series(
                train_set[split_idx:, 1], sequence_len, input_dim, output_dim_val
            )

            if train_x is None:
                train_x = formatted_train_set[0]
                train_y = formatted_train_set[1]
            else:
                train_x = np.concatenate((train_x, formatted_train_set[0]), axis=0)
                train_y = np.concatenate((train_y, formatted_train_set[1]), axis=0)

            if val_x is None:
                val_x = formatted_val_set[0]
                val_y = formatted_val_set[1]
            else:
                val_x = np.concatenate((val_x, formatted_val_set[0]), axis=0)
                val_y = np.concatenate((val_y, formatted_val_set[1]), axis=0)

        if zero_centre:
            print("Zero centering the data...")
            self.train_mean = np.mean(complete_train_set)
            train_x -= self.train_mean
            train_y -= self.train_mean
            val_x -= self.train_mean
            val_y -= self.train_mean
            print("Done.")

        if normalize:
            print("Normalizing the data...")
            self.train_std_dev = np.std(complete_train_set)
            train_x /= self.train_std_dev
            train_y /= self.train_std_dev
            val_x /= self.train_std_dev
            val_y /= self.train_std_dev
            print("Done.")

        if min_zero_max_one:
            print("Squashing the data between 0 and 1...")
            train_x = util.sigmoid(train_x)
            train_y = util.sigmoid(train_y)
            val_x = util.sigmoid(val_x)
            val_y = util.sigmoid(val_y)
            self.squashed = True
            print("Done.")

        return (train_x, train_y), (val_x, val_y)

    def restore_data(self, data):
        """
        Restore data to its original form by applying inverse of data pre processing operations. 
        """
        if self.squashed:
            data = util.logit(data)
        if self.train_std_dev is not None:
            data *= self.train_std_dev
        if self.train_mean is not None:
            data += self.train_mean
        return data


def unpickle(filename):
    try:
        with open(filename, 'rb') as fo:
            data_obj = cPickle.load(fo)
    except Exception as e:
        print("Data not found! Please make sure you have all data files in " +
              os.path.abspath('../data'))
        data_obj = None
    return data_obj
