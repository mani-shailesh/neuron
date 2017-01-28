import cPickle
import os
import random

DATA_DIR = "../data"
NUM_TRAIN_FILES = 5


class DataProcessor:
    """
    Class to handle all raw data related operations
    """
    def __init__(self, data_dir=DATA_DIR, num_train_files=NUM_TRAIN_FILES):
        self.data_dir = data_dir
        self.num_train_files = num_train_files
        """
        Number of files for training data
        """
        self.label_data_dict = {}
        """
        Dictionary with label indices as keys.
        Each value is a dictionary with following keys:
        'name', 'train_instance_list', 'validation_instance_list' & 'test_instance_list'
        """

    def load_data(self):
        """
        Loads training and test data from the `data_dir` and resets 'label_data_dict'
        """
        self.label_data_dict = {}
        meta_filename = "batches.meta"
        label_name_list = unpickle(os.path.join(self.data_dir, meta_filename))['label_names']
        for ii, label_name in enumerate(label_name_list):
            self.label_data_dict[ii] = {
                'name': label_name,
                'train_instance_list':  [],
                'validation_instance_list': [],
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

    def create_validation_set(self, validation_fraction=0.2):
        """
        Splits the training data loaded in 'label_data_list' into training and validation
        Updates 'label_data_dict' accordingly
        """
        for label in self.label_data_dict.keys():
            train_instance_list = self.label_data_dict[label]['train_instance_list']
            random.shuffle(train_instance_list)
            num_validation = int(len(train_instance_list) * validation_fraction)
            self.label_data_dict[label]['validation_instance_list'] = train_instance_list[0:num_validation]
            self.label_data_dict[label]['train_instance_list'] = train_instance_list[num_validation:]


def unpickle(filename):
    fo = open(filename, 'rb')
    data_obj = cPickle.load(fo)
    fo.close()
    return data_obj
