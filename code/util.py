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
