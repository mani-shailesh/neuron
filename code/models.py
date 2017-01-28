import util


class RMLR:
    """
    Classifier that uses Regularized Multinomial Logistic Regression
    """
    def __init__(self):
        self.weights = None
        pass

    def train(self, X, Y, lr, batch_size, num_epochs, val_X=None, val_Y=None):
        """
        Train the model using 'X' as training data and 'Y' as labels
        :param X: numpy array of feature vectors
        :param Y: list of actuals label indices
        :param lr:  learning rate float (> 0)
        :param batch_size:  int (> 0)
        :param num_epochs:  int (>= 0)
        :param val_X:   numpy array of feature vectors for validation set
                        No validation performed if this is None
        :param val_Y:   class labels for validation set
                        Used only if val_X is not None
        """
        pass

    def predict(self, X):
        """
        Predict the class labels using 'X' as test data
        :param X: numpy array of feature vectors
        :return: list of predicted class label indices
        """
        pass
