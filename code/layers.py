import numpy as np

import util


class Layer:
    """
    Base class for a layer
    """

    def __init__(self, name, input_layer=None, input_shape=None):
        """
        Initialization of this layer
        :param name: Name of the layer. Must be unique
        :param input_layer: Input layer to this layer
        :param input_shape: Shape of input to this layer. Used iff input_layer is None.
        One of input_layer or input_shape must be specified
        """
        self.input_layer = input_layer
        if self.input_layer is None:
            self.input_shape = input_shape
        else:
            self.input_shape = self.input_layer.get_output_shape()
        self.name = name

    def get_output_shape(self):
        pass

    def forward_pass(self, X):
        pass

    def back_propagation(self):
        pass


class Dense(Layer):
    """
    Fully connected layer
    """

    def __init__(self, num_units, *args, **kwargs):
        """
        Initialization of fully connected layer
        :param num_units: Number of units in this layer
        """
        Layer.__init__(self, *args, **kwargs)
        self.h = num_units
        weight_shape = (self.h, self.input_shape[1] + 1)
        self.w = np.zeros(weight_shape)

    def get_output_shape(self):
        """
        Return the output shape of this layer
        :return: tuple
        """
        output_shape = (self.input_shape[0], self.h)
        return output_shape

    def forward_pass(self, X):
        """
        Perform forward pass on the input
        :param X: input numpy array
        :return: numpy array
        """
        X = util.add_dummy_feature(X)
        return np.dot(X, np.transpose(self.w))


class Activation(Layer):
    """
    Base class for activation layers
    """

    def get_output_shape(self):
        """
        Return the output shape of this layer
        :return: tuple
        """
        output_shape = self.input_shape
        return output_shape


class Softmax(Activation):
    """
    Softmax activation layer
    """

    def forward_pass(self, X):
        """
        Perform forward pass on the input
        :param X: input numpy array
        :return: numpy array of shape same as input
        """
        row_sum = np.sum(np.exp(X), axis=1)
        return np.exp(X) / row_sum[:, None]


class Sigmoid(Activation):
    """
    Sigmoid activation layer
    """

    def forward_pass(self, X):
        """
        Perform forward pass on the input
        :param X: input numpy array
        :return: numpy array of shape same as input
        """
        return 1 / (1 + np.exp(-1 * X))


class ReLU(Activation):
    """
    ReLU activation layer
    """

    def forward_pass(self, X):
        """
        Perform forward pass on the input
        :param X: input numpy array
        :return: numpy array of shape same as input
        """
        indices = X < np.zeros(self.input_shape)
        X[indices] = 0
        return X


class LeakyReLU(Activation):
    """
    LeakyReLU activation layer
    """

    def __init__(self, alpha=0.3, *args, **kwargs):
        Activation.__init__(self, *args, **kwargs)
        self.alpha = alpha

    def forward_pass(self, X):
        """
        Perform forward pass on the input
        :param X: input numpy array
        :return: numpy array of shape same as input
        """
        indices = X < np.zeros(self.input_shape)
        X[indices] *= self.alpha
        return X


class Tanh(Activation):
    """
    Tanh activation layer
    """

    def forward_pass(self, X):
        """
        Perform forward pass on the input
        :param X: input numpy array
        :return: numpy array of shape same as input
        """
        return np.tanh(X)
