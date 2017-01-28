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
        self.validation_fraction = validation_fraction
        """
        Fraction of training data to be split and used as validation set
        """

    def load_data(self):
        """
        Loads training and test data from the `data_dir` and resets 'data_dict' & 'label_details_dict'
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

        label_data_dict = self.create_validation_set(label_data_dict)

        key_list = ['train', 'val', 'test']
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
        if self.data_dict is None:
            self.load_data()

        key_list = ['train', 'val', 'test']

        if zero_centre:
            print("Zero centering the data...")
            train_mean = np.mean(self.data_dict['train']['data'], axis=0)
            for key in key_list:
                if len(self.data_dict[key]['data']) != 0:
                    self.data_dict[key]['data'] -= train_mean
            print("Done.")

        if normalize:
            print("Normalizing the data...")
            train_std_dev = np.std(self.data_dict['train']['data'], axis=0)
            for key in key_list:
                self.data_dict[key]['data'] /= train_std_dev
            print("Done.")

        return self.data_dict


def unpickle(filename):
    fo = open(filename, 'rb')
    data_obj = cPickle.load(fo)
    fo.close()
    return data_obj
