import numpy as np


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
