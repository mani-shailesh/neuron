import cPickle
import os
import random
import numpy as np

DATA_DIR = "../data"
NUM_TRAIN_FILES = 5


class DataStore:
    """
    Class to handle all raw data related operations
    """
    def __init__(self, data_dir=DATA_DIR, num_train_files=NUM_TRAIN_FILES, validation_fraction=0.2):
        self.data_dir = data_dir
        self.num_train_files = num_train_files
        """
        Number of files for training data
        """
        self.label_data_dict = None
        """
        Dictionary with label indices as keys.
        Each value is a dictionary with following keys:
        'name', 'train_instance_list', 'val_instance_list' & 'test_instance_list'
        """
        self.validation_fraction = validation_fraction
        """
        Fraction of training data to be split and used as validation set
        """

    def load_data(self):
        """
        Loads training and test data from the `data_dir` and resets 'label_data_dict'
        """
        print("Loading data...")

        self.label_data_dict = {}
        meta_filename = "batches.meta"
        label_name_list = unpickle(os.path.join(self.data_dir, meta_filename))['label_names']
        for ii, label_name in enumerate(label_name_list):
            self.label_data_dict[ii] = {
                'name': label_name,
                'train_instance_list':  [],
                'val_instance_list': [],
                'test_instance_list':   []
            }

        base_filename = "data_batch_"
        for ii in range(1, self.num_train_files+1):
            filename = base_filename + str(ii)
            data_dict = unpickle(os.path.join(self.data_dir, filename))
            for jj, label_idx in enumerate(data_dict['labels']):
                self.label_data_dict[label_idx]['train_instance_list'].append(data_dict['data'][jj])

        test_filename = "test_batch"
        data_dict = unpickle(os.path.join(self.data_dir, test_filename))
        for jj, label_idx in enumerate(data_dict['labels']):
            self.label_data_dict[label_idx]['test_instance_list'].append(data_dict['data'][jj])

        print("Done.")

    def create_validation_set(self):
        """
        Splits the training data loaded in 'label_data_list' into training and validation
        Updates 'label_data_dict' accordingly
        """
        print("Splitting training set into training and validation sets...")
        for label in self.label_data_dict.keys():
            train_instance_list = self.label_data_dict[label]['train_instance_list']
            random.shuffle(train_instance_list)
            num_validation = int(len(train_instance_list) * self.validation_fraction)
            self.label_data_dict[label]['val_instance_list'] = train_instance_list[0:num_validation]
            self.label_data_dict[label]['train_instance_list'] = train_instance_list[num_validation:]
        print("Done.")

    def get_data(self, zero_centre=True, normalize=True):
        """
        Return training, validation and test data with or without pre-processing
        :param zero_centre: Perform zero centering on data if this is True
        :param normalize: Normalize the data if this is True
        :return: A dictionary with 'train', 'val' & 'test' as keys
        Each value in this dictionary in turn is a dictionary with following structure
        'data': numpy array of data
        'labels':   numpy array of labels in same order as data
        """

        # Make sure that data is loaded
        if self.label_data_dict is None:
            self.load_data()
            self.create_validation_set()

        key_list = ['train', 'val', 'test']
        data_dict = {
            key:    {'data':    [], 'labels':   []} for key in key_list
        }

        # populate data_dict
        for label_idx in self.label_data_dict.keys():
            for key in key_list:
                data_dict[key]['data'].extend(self.label_data_dict[label_idx][key + '_instance_list'])
                data_dict[key]['labels'].extend(
                    [label_idx for _ in range(len(self.label_data_dict[label_idx][key + '_instance_list']))]
                )

        for key in key_list:
            data_dict[key]['data'] = np.array(data_dict[key]['data'], dtype=np.float)
            data_dict[key]['labels'] = np.array(data_dict[key]['labels'], dtype=np.float)

        if zero_centre:
            print("Zero centering the data...")
            train_mean = np.mean(data_dict['train']['data'], axis=0)
            for key in key_list:
                if len(data_dict[key]['data']) != 0:
                    data_dict[key]['data'] -= train_mean
            print("Done.")

        if normalize:
            print("Normalizing the data...")
            train_std_dev = np.std(data_dict['train']['data'], axis=0)
            for key in key_list:
                data_dict[key]['data'] /= train_std_dev
            print("Done.")

        return data_dict


def unpickle(filename):
    fo = open(filename, 'rb')
    data_obj = cPickle.load(fo)
    fo.close()
    return data_obj
