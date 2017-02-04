import json
import os

import numpy as np
from tqdm import tqdm

import layers
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

    def train(self, X, Y, lr, batch_size, num_epochs, lambda_, val_X=None, val_Y=None,
              reinit_weights=False, print_acc=True):
        """
        Train the model using 'X' as training data and 'Y' as labels
        :param print_acc: Print to console if True
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
            self.W = np.random.standard_normal(w_shape) / np.sqrt(d)
            self.W[:, 0] = 0

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

            for init_idx in tqdm(range(0, n, batch_size)):
                y = Y[init_idx:init_idx + batch_size]
                x = X[init_idx:init_idx + batch_size, :]
                y_c = util.to_categorical(y, self.num_classes)
                f = self.predict_values(x)
                # noinspection PyTypeChecker
                del_J = np.dot(np.transpose(f - y_c), x) / batch_size + \
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

            if print_acc:
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

    def __init__(self, input_layer, output_layer, loss, name='model', log_file=None):
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
        if type(loss) is str:
            self.loss = getattr(layers, loss)()
        else:
            self.loss = loss
        self.name = name
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

            for init_idx in tqdm(range(0, n, batch_size)):
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

    def save_model(self, save_dir, save_weights=True):
        """
        Save the current architecture and weights of model in given `save_dir`
        :param save_dir: The directory to save the model in
        :param save_weights: Save weights as well is this is True
        :return:
        """

        json_file = os.path.join(save_dir, self.name + '.json')
        json_dict = {
            'name': self.name,
            'loss': self.loss.__class__.__name__,
            'layers': [],
            'input_layer_name': self.layers[0].get_name(),
            'output_layer_name': self.layers[-1].get_name()
        }
        if self.log_file is not None:
            json_dict['log_file'] = self.log_file
        for layer in self.layers:
            json_dict['layers'].append(layer.get_config())
        json_str = json.dumps(json_dict)
        with open(json_file, 'w') as json_file_object:
            json_file_object.write(json_str)

        if save_weights:
            weights_file = os.path.join(save_dir, self.name + '_weights.hdf5')
            for layer in self.layers:
                layer.save_weights(weights_file)

    @staticmethod
    def load_from_json(json_file_name):
        """
        Load a model from `json_file_name` and return an instance of this class
        :param json_file_name: Full path to the json file.
        :return: `MLP` instance
        """
        with open(json_file_name, 'r') as json_file_obj:
            loaded_dict = json.load(json_file_obj)
        layer_name_to_obj = {}
        for layer_dict in loaded_dict['layers']:
            layer_class = getattr(layers, layer_dict['type'])
            if 'input_layer_name' in layer_dict:
                layer_dict['input_layer'] = layer_name_to_obj[layer_dict['input_layer_name']]
            layer_obj = layer_class(**layer_dict)
            layer_name_to_obj[layer_obj.get_name()] = layer_obj
        loaded_dict['input_layer'] = layer_name_to_obj[loaded_dict['input_layer_name']]
        loaded_dict['output_layer'] = layer_name_to_obj[loaded_dict['output_layer_name']]

        return MLP(**loaded_dict)
