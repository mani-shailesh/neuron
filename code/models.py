import numpy as np
from tqdm import tqdm

import util


# noinspection PyPep8Naming
class RMLR:
    """
    Classifier that uses Regularized Multinomial Logistic Regression
    """
    def __init__(self, num_classes, log_filename=None):
        self.W = None
        self.num_classes = num_classes
        self.log_file = log_filename
        """
        Weights including bias for the model
        """
        pass

    def train(self, X, Y, lr, batch_size, num_epochs, lambda_, val_X=None, val_Y=None, reinit_weights=False):
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
        :param reinit_weights: Reinitialize weights iff this is True or weights are None
        :return Tuple of best validation accuracy, training accuracy at that epoch and the epoch no.
        """

        # Add dummy features for bias terms
        X = util.add_dummy_feature(X)
        val_X = util.add_dummy_feature(val_X)

        n = X.shape[0]
        d = X.shape[1]

        # Initialize the weights
        if reinit_weights or self.W is None:
            w_shape = (self.num_classes, d)
            self.W = np.zeros(w_shape)

        print("Starting training...")

        best_val_acc = 0
        best_train_acc = 0
        best_epoch = 1

        util.log(self.log_file, "epoch,train_acc,val_acc")

        for epoch_idx in range(num_epochs):
            print("Performing epoch #" + str(epoch_idx + 1))

            # Shuffle training data and labels
            perm = np.random.permutation(n)
            X = X[perm, :]
            Y = Y[perm]

            for init_idx in tqdm(range(0, n, batch_size)):
                y = Y[init_idx:init_idx + batch_size]
                x = X[init_idx:init_idx + batch_size, :]
                y_c = util.to_categorical(y, self.num_classes)
                f = self.predict_values(x)
                # noinspection PyTypeChecker
                del_J = np.dot(np.transpose(f - y_c), x) + \
                        (lambda_ * np.c_[np.zeros(self.num_classes), self.W[:, 1:]])
                self.W -= (lr * del_J)

            train_acc = util.get_accuracy(Y, self.predict_classes(X))
            val_acc = util.get_accuracy(val_Y, self.predict_classes(val_X))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_train_acc = train_acc
                best_epoch = epoch_idx + 1

            util.log(self.log_file,
                     str(epoch_idx + 1) + "," + str(train_acc) + "," + str(val_acc))
            print("Training accuracy: " + str(train_acc))
            print("Validation accuracy: " + str(val_acc))
            print("\n------------------------------------\n")

        print("Training completed.")
        return best_train_acc, best_val_acc, best_epoch

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
        # noinspection PyTypeChecker
        return np.argmax(values, axis=1)


class MLP:
    """
    Class to represent a multi layer perceptron model composed of layers
    """

    def __init__(self, input_layer, output_layer, loss, log_file=None):
        """
        Initialize the model
        :param input_layer: 'Layer' object to act as the input layer
        :param output_layer: 'Layer' object to act as the output layer
        :param loss:    Object of `Loss` layer class
        """
        self.layers = [output_layer]
        layer = output_layer
        while layer is not input_layer:
            layer = layer.get_input_layer()
            if layer is None:
                print("Can not find connection from input layer to output layer!")
                self.layers = []
                return
            else:
                self.layers.append(layer)
        self.layers.reverse()
        self.loss = loss
        self.log_file = log_file
        self.num_classes = output_layer.get_output_shape()[1]

    def train(self, X, Y, lr, batch_size, num_epochs, val_X, val_Y, print_acc=True):
        """
        Train the model using 'X' as training data and 'Y' as labels
        :param print_acc: Print accuracy after each epoch if True
        :param X: numpy array of feature vectors
        :param Y: list of actuals label indices
        :param lr:  learning rate float (> 0)
        :param batch_size:  int (> 0)
        :param num_epochs:  int (>= 0)
        :param val_X:   numpy array of feature vectors for validation set
                        No validation performed if this is None
        :param val_Y:   class labels for validation set
                        Used only if val_X is not None
        :return Tuple of best validation accuracy, training accuracy at that epoch and the epoch no.
        """

        n = X.shape[0]

        print("Starting training...")

        best_val_acc = 0
        best_train_acc = 0
        best_epoch = 1

        util.log(self.log_file, "epoch,train_acc,val_acc")

        for epoch_idx in tqdm(range(num_epochs)):
            if print_acc:
                print("Performing epoch #" + str(epoch_idx + 1))

            # Shuffle training data and labels
            perm = np.random.permutation(n)
            X = X[perm, :]
            Y = Y[perm]

            for init_idx in range(0, n, batch_size):
                y = Y[init_idx:init_idx + batch_size]
                x = X[init_idx:init_idx + batch_size, :]
                y_c = util.to_categorical(y, self.num_classes)
                o = self.forward_pass(x)

                # Perform back propagation on the network
                grad = self.loss.get_gradient(y_c, o)
                for layer in reversed(self.layers):
                    grad = layer.back_propagation(grad, lr)

            train_acc = util.get_accuracy(Y, self.predict_classes(X))
            val_acc = util.get_accuracy(val_Y, self.predict_classes(val_X))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_train_acc = train_acc
                best_epoch = epoch_idx + 1

            util.log(self.log_file,
                     str(epoch_idx + 1) + "," + str(train_acc) + "," + str(val_acc))
            if print_acc:
                print("Training accuracy: " + str(train_acc))
                print("Validation accuracy: " + str(val_acc))
                print("\n------------------------------------\n")

        print("Training completed.")
        return best_train_acc, best_val_acc, best_epoch

    def forward_pass(self, X):
        """
        Perform forward pass of the network on given data
        :param X: numpy array of input data N x D
        :return: numpy array of output N x C
        """
        for layer in self.layers:
            X = layer.forward_pass(X)
        return X

    def predict_classes(self, X):
        """
        Predict the class labels using 'X' as test data
        :param X: numpy array of feature vectors
        :return: list of predicted class label indices
        """
        values = self.forward_pass(X)
        return np.argmax(values, axis=1)
