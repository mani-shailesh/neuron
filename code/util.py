import numpy as np
import pandas as pd


def add_dummy_feature(X):
    """
    Adds a dummy feature with all 1s as the first feature of X
    Useful for adding a feature for bias operations
    :param X: numpy array of feature vectors
    :return updated X
    """
    X = np.c_[np.ones(len(X)), X]
    return X


def get_accuracy(original, predicted):
    """
    Return percentage accuracy of predicted class labels
    :param original: Original class labels
    :param predicted: Predicted class labels
    :return: float
    """
    return (np.sum(np.equal(original, predicted)) * 100.0)/len(original)


def to_categorical(Y, num_classes):
    """
    Convert class indices to categorical boolean vectors
    :param Y: Numpy array of class indices (N x 1)
    :param num_classes: Number of total classes
    :return: (N x num_classes) boolean matrix (numpy array)
    """
    n = len(Y)
    yc = np.zeros((n, num_classes))
    yc[range(n), Y] = 1
    return yc


def log(filename, msg):
    """
    Writes log 'msg' to 'filename' and appends a newline
    :param filename: string - path to file
    :param msg: string - message to log
    """
    if filename is not None:
        with open(filename, 'a') as logfile:
            logfile.write(str(msg) + "\n")


def get_lists_from_csv(filename):
    """
    Returns dict with file headers, list of values as key, value pairs
    :param filename: Full path of the csv file
    :return: dict {'header1': [value1, value2, ...]}
    """
    data = pd.read_csv(filename)
    header_list = list(data)
    ret_dict = {key: [] for key in header_list}
    for index, row in data.iterrows():
        for header in header_list:
            ret_dict[header].append(row[header])
    return ret_dict


def sigmoid(X):
    """
    :param X: np array
    :return: sigmoid(X)
    """
    return 1 / (1 + np.exp(-1 * X))


def logit(X):
    """
    :param X: np array
    :return: logit(X)
    """
    return np.log(X / (1 - X))


def mse(Y, O):
    """
    Calculate mean squared error between Y and O
    :param Y: 1-D Array
    :param O: 1-D Array
    """
    return np.mean(np.power((Y - O), 2))


def convert_to_time_series(dataset, sequence_len, input_dim, output_dim):
    """
    Function to convert a sequential `dataset` into time series data
    :param dataset: 1-D List containing sequential data
    :param sequence_len: Number of time steps in each instance
    :param input_dim: Dimension of input at each time step
    :param output_dim: Dimension of output for each instance
    :return: tuple containing : 
    numpy array of shape (None, sequence_len, input_dim), numpy array of shape (None, output_dim)
    """
    data_x, data_y = [], []
    for ii in range(0, len(dataset) - sequence_len * input_dim - output_dim + 1, input_dim):
        data_x.append(np.reshape(dataset[ii:ii + sequence_len * input_dim], (sequence_len, input_dim)))
        data_y.append(dataset[ii + sequence_len * input_dim:ii + sequence_len * input_dim + output_dim])
    return np.array(data_x), np.array(data_y)


def get_predictions(model, val_x, out_dim):
    if len(val_x) == 0:
        print('No predictions for empty input!')
        raise Exception

    model_out_dim = model.layers[-1].get_output_shape()[1]
    if out_dim % model_out_dim != 0:
        print('out_dim in not a multiple of model_out_dim!')
        raise Exception

    final_predictions = np.zeros((val_x.shape[0], out_dim))
    new_val_x = np.copy(val_x)
    for ii in range(out_dim / model_out_dim):
        predictions = model.forward_pass(new_val_x)
        final_predictions[:, ii * model_out_dim : (ii + 1) * model_out_dim] = predictions

        for idx in range(new_val_x.shape[0]):
            row = np.reshape(new_val_x[idx], np.product(new_val_x[idx].shape))
            row[0:-1 * model_out_dim] = row[model_out_dim:]
            row[-1 * model_out_dim:] = predictions[idx]
            new_val_x[idx, :, :] = np.reshape(row, new_val_x[idx].shape)

    return final_predictions
