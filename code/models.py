import json
import os

import h5py
import numpy as np
from tqdm import tqdm

import layers
import util


# noinspection PyPep8Naming
class RMLR:
    """
    Classifier that uses Regularized Multinomial Logistic Regression
    """
    def __init__(self, num_classes, weight_decay, name='model_rmlr', log_file=None):
        self.W = None
        """
        Weights including bias for the model
        """
        self.name = name
        self.num_classes = num_classes
        self.log_file = log_file
        self.weight_decay = weight_decay

    def train(self, X, Y, lr, batch_size, num_epochs, val_X=None, val_Y=None,
              print_acc=True, save_dir=None):
        """
        Train the model using 'X' as training data and 'Y' as labels
        :param print_acc: Print to console if True
        :param X: numpy array of feature vectors
        :param Y: list of actuals label indices
        :param lr:  learning rate float (> 0)
        :param batch_size:  int (> 0)
        :param num_epochs:  int (>= 0)
        :param val_X:   numpy array of feature vectors for validation set
                        No validation performed if this is None
        :param val_Y:   class labels for validation set
                        Used only if val_X is not None
        :param save_dir: Save best model if this is not None
        :return Tuple of best validation accuracy, training accuracy at that epoch and the epoch no.
        """

        # Add dummy features for bias terms
        X = util.add_dummy_feature(X)

        if save_dir is not None:
            json_file = os.path.join(save_dir, self.name + '.json')
            self.save_model_json(json_file)
            weights_file = os.path.join(save_dir, self.name + '_weights.hdf5')

        n = X.shape[0]
        d = X.shape[1]

        # Initialize the weights
        if self.W is None:
            w_shape = (self.num_classes, d)
            # self.W = np.random.standard_normal(w_shape) / np.sqrt(d)
            # self.W[:, 0] = 0
            self.W = np.zeros(w_shape)

        print("Starting training...")

        best_val_acc = 0

        util.log(self.log_file, "epoch,train_acc,val_acc")

        train_acc_list = []
        val_acc_list = []

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
                f = self.predict_values(x, False)
                # noinspection PyTypeChecker
                del_J = np.dot(np.transpose(f - y_c), x) / batch_size + \
                        (self.weight_decay * np.c_[np.zeros(self.num_classes), self.W[:, 1:]])
                self.W -= (lr * del_J)

            train_acc = util.get_accuracy(Y, self.predict_classes(X, False))
            train_acc_list.append(train_acc)

            log_str = str(epoch_idx + 1) + "," + str(train_acc)
            if print_acc:
                print("Training accuracy: " + str(train_acc))

            if val_X is not None and val_Y is not None:
                val_acc = util.get_accuracy(val_Y, self.predict_classes(val_X))
                val_acc_list.append(val_acc)

                log_str = log_str + "," + str(val_acc)
                if print_acc:
                    print("Validation accuracy: " + str(val_acc))
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    if save_dir is not None:
                        self.save_model_weights(weights_file)

            util.log(self.log_file, log_str)
            if print_acc:
                print("\n------------------------------------\n")

        print("Training completed.")

        if save_dir is not None and (val_X is None or val_Y is None):
            print("Saving model...")
            self.save_model_weights(weights_file)
            print("Done.")

        return train_acc_list, val_acc_list

    def predict_values(self, X, add_dummy_param=True):
        """
        Predict outputs for each regressor
        :param X: np array with shape(N, D)
        :param add_dummy_param: Add dummy parameter if this is true
                                else assume it is already present
        :return: output np array with shape (N, C)
        """
        if add_dummy_param:
            X = util.add_dummy_feature(X)
        f = np.dot(X, np.transpose(self.W))
        return 1 / (1 + np.exp(-1 * f))

    def predict_classes(self, X, add_dummy_param=True):
        """
        Predict the class labels using 'X' as test data
        :param X: numpy array of feature vectors
        :param add_dummy_param: Add dummy parameter if this is true
                                else assume it is already present
        :return: list of predicted class label indices
        """
        values = self.predict_values(X, add_dummy_param)
        # noinspection PyTypeChecker
        return np.argmax(values, axis=1)

    def save_model_json(self, json_file):
        """
        Save the current architecture in given path
        :param json_file: Full path of the file to write to
        :return:
        """
        json_dict = {
            'name': self.name,
            'weight_decay': self.weight_decay,
            'num_classes':  self.num_classes
        }
        if self.log_file is not None:
            json_dict['log_file'] = self.log_file
        json_str = json.dumps(json_dict)
        with open(json_file, 'w') as json_file_object:
            json_file_object.write(json_str)

    def save_model_weights(self, weights_file):
        """
        Save the current weights in given path
        :param weights_file: Full path of the file to write to
        :return:
        """
        with h5py.File(weights_file, 'w') as w_file_obj:
            w_file_obj.create_dataset('w', self.W.shape, 'f', self.W)

    def save_model(self, save_dir, save_weights=True):
        """
        Save the current architecture and weights of model in given `save_dir`
        :param save_dir: The directory to save the model in
        :param save_weights: Save weights as well is this is True
        :return:
        """

        json_file = os.path.join(save_dir, self.name + '.json')
        self.save_model_json(json_file)

        if save_weights:
            weights_file = os.path.join(save_dir, self.name + '_weights.hdf5')
            self.save_model_weights(weights_file)

    @staticmethod
    def load_from_json(json_file_name):
        """
        Load a model from `json_file_name` and return an instance of this class
        :param json_file_name: Full path to the json file.
        :return: `RMLR` instance
        """
        with open(json_file_name, 'r') as json_file_obj:
            loaded_dict = json.load(json_file_obj)

        return RMLR(**loaded_dict)

    def load_weights(self, weights_file_name):
        """
        Load weights to this model
        :param weights_file_name: Full path to the file containing weights
        :return:
        """
        with h5py.File(weights_file_name, 'r') as w_file:
            self.W = w_file['w'][...]

    @staticmethod
    def load_model(json_file_name, weights_file_name=None):
        """
        Load and return model and its weights(if given)
        :param json_file_name: Full path to the json file.
        :param weights_file_name: Full path to the file containing weights
        :return: `RMLR` instance
        """
        model = RMLR.load_from_json(json_file_name)
        if weights_file_name is not None:
            model.load_weights(weights_file_name)
        return model


class MLP:
    """
    Class to represent a multi layer perceptron model composed of layers
    """

    def __init__(self, input_layer, output_layer, loss, name='model_mlp', log_file=None,
                 *args, **kwargs):
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

    def train(self, X, Y, lr, batch_size, num_epochs, is_classification=False, val_X=None, val_Y=None
              , print_loss=True, save_dir=None):
        """
        Train the model using 'X' as training data and 'Y' as labels
        :param print_loss: Print loss after each epoch if True
        :param X: numpy array of feature vectors
        :param Y: list of actuals label indices
        :param lr:  learning rate float (> 0)
        :param batch_size:  int (> 0)
        :param num_epochs:  int (>= 0)
        :param is_classification: True is the task is that of classification
        :param val_X:   numpy array of feature vectors for validation set
                        No validation performed if this is None
        :param val_Y:   class labels for validation set
                        Used only if val_X is not None
        :param save_dir: Save best model if this is not None
        :return List of training losses and validation losses after each epoch
        """

        if save_dir is not None:
            json_file = os.path.join(save_dir, self.name + '.json')
            self.save_model_json(json_file)
            weights_file = os.path.join(save_dir, self.name + '_weights.hdf5')

        n = X.shape[0]

        print("Starting training...")

        best_val_loss = float('inf')

        util.log(self.log_file, "epoch,train_loss,val_loss")

        train_loss_list = []
        val_loss_list = []

        for epoch_idx in tqdm(range(num_epochs)):
            if print_loss:
                print("Performing epoch #" + str(epoch_idx + 1))

            # Shuffle training data and labels
            perm = np.random.permutation(n)
            X = X[perm]
            Y = Y[perm]

            if is_classification:
                Y = util.to_categorical(Y, self.num_classes)
                if val_Y is not None:
                    val_Y = util.to_categorical(val_Y, self.num_classes)

            for init_idx in tqdm(range(0, n, batch_size)):
                y = Y[init_idx:init_idx + batch_size]
                x = X[init_idx:init_idx + batch_size]

                o = self.forward_pass(x)

                # Perform back propagation on the network
                grad = self.loss.get_gradient(y, o)
                for layer in reversed(self.layers):
                    grad = layer.back_propagation(grad, lr)

            train_loss = np.mean(self.loss.get_loss_value(Y, self.forward_pass(X)))
            train_loss_list.append(train_loss)

            log_str = str(epoch_idx + 1) + "," + str(train_loss)
            if print_loss:
                print("Training loss: " + str(train_loss))

            if val_X is not None and val_Y is not None and len(val_X) > 0 and len(val_Y) > 0:
                val_loss = np.mean(self.loss.get_loss_value(val_Y, util.get_predictions(self, val_X, val_Y.shape[1])))
                val_loss_list.append(val_loss)

                log_str = log_str + "," + str(val_loss)
                if print_loss:
                    print("Validation loss: " + str(val_loss))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if save_dir is not None:
                        self.save_model_weights(weights_file)

            util.log(self.log_file, log_str)
            if print_loss:
                print("\n------------------------------------\n")

        print("Training completed.")
        if save_dir is not None and (val_X is None or val_Y is None):
            print("Saving model...")
            self.save_model_weights(weights_file)
            print("Done.")
        return train_loss_list, val_loss_list

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

    def save_model_json(self, json_file):
        """
        Save the current architecture in given path
        :param json_file: Full path of the file to write to
        :return:
        """
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

    def save_model_weights(self, weights_file):
        """
        Save the current weights in given path
        :param weights_file: Full path of the file to write to
        :return:
        """
        with h5py.File(weights_file, 'w') as weights_file_obj:
            for layer in self.layers:
                layer.save_weights(weights_file_obj)

    def save_model(self, save_dir, save_weights=True):
        """
        Save the current architecture and weights of model in given `save_dir`
        :param save_dir: The directory to save the model in
        :param save_weights: Save weights as well is this is True
        :return:
        """

        json_file = os.path.join(save_dir, self.name + '.json')
        self.save_model_json(json_file)

        if save_weights:
            weights_file = os.path.join(save_dir, self.name + '_weights.hdf5')
            self.save_model_weights(weights_file)

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

    def load_weights(self, weights_file_name):
        """
        Load weights to this model
        :param weights_file_name: Full path to the file containing weights
        :return:
        """
        for layer in self.layers:
            layer.load_weights(weights_file_name)

    @staticmethod
    def load_model(json_file_name, weights_file_name=None):
        """
        Load and return model and its weights(if given)
        :param json_file_name: Full path to the json file.
        :param weights_file_name: Full path to the file containing weights
        :return: `MLP` instance
        """
        model = MLP.load_from_json(json_file_name)
        if weights_file_name is not None:
            model.load_weights(weights_file_name)
        return model
