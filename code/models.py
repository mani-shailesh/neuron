import util
import numpy as np


# noinspection PyPep8Naming
class RMLR:
    """
    Classifier that uses Regularized Multinomial Logistic Regression
    """
    def __init__(self, num_classes, log_filename):
        self.W = None
        self.num_classes = num_classes
        self.log_file = log_filename
        """
        Weights including bias for the model
        """
        pass

    def train(self, X, Y, lr, batch_size, num_epochs, lambda_, val_X=None, val_Y=None):
        """
        Train the model using 'X' as training data and 'Y' as labels
        :param X: numpy array of feature vectors
        :param Y: list of actuals label indices
        :param lr:  learning rate float (> 0)
        :param batch_size:  int (> 0)
        :param num_epochs:  int (>= 0)
        :param lambda_: regularization parameter
        :param val_X:   numpy array of feature vectors for validation set
                        No validation performed if this is None
        :param val_Y:   class labels for validation set
                        Used only if val_X is not None
        """

        # Add dummy features for bias terms
        X = util.add_dummy_feature(X)
        val_X = util.add_dummy_feature(val_X)

        n = X.shape[0]
        d = X.shape[1]

        # Initialize the weights
        w_shape = (self.num_classes, d)
        self.W = np.zeros(w_shape)

        print("Starting training...")

        util.log(self.log_file, "epoch,train_acc,val_acc")

        for epoch_idx in range(num_epochs):
            print("Performing epoch #" + str(epoch_idx + 1))

            # Shuffle training data and labels
            perm = np.random.permutation(n)
            X = X[perm, :]
            Y = Y[perm]

            for init_idx in range(0, n, batch_size):
                y = Y[init_idx:init_idx + batch_size]
                x = X[init_idx:init_idx + batch_size, :]
                y_c = util.to_categorical(y, self.num_classes)
                f = self.predict_values(x)
                del_J = np.dot(np.transpose(f - y_c), x) + \
                        (lambda_ * np.c_[np.zeros(self.num_classes), self.W[:, 1:]])
                self.W -= (lr * del_J)

            train_acc = util.get_accuracy(Y, self.predict_classes(X))
            val_acc = util.get_accuracy(val_Y, self.predict_classes(val_X))

            util.log(self.log_file,
                     str(epoch_idx + 1) + "," + str(train_acc) + "," + str(val_acc))
            print("Training accuracy: " + str(train_acc))
            print("Validation accuracy: " + str(val_acc))
            print("\n------------------------------------\n")

        print("Training completed.")

    def predict_values(self, X):
        """
        Predict outputs for each regressor
        :param X: np array with shape(N, D)
        :return: output np array with shape (N, C)
        """
        f = np.dot(X, np.transpose(self.W))
        return 1 / (1 + np.exp(-1 * f))

    def predict_classes(self, X):
        """
        Predict the class labels using 'X' as test data
        :param X: numpy array of feature vectors
        :return: list of predicted class label indices
        """
        values = self.predict_values(X)
        return np.argmax(values, axis=1)
