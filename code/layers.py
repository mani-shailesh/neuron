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

    def back_propagation(self, f, d):
        pass

    def get_input_layer(self):
        """
        :return: input layer to this layer
        """
        return self.input_layer


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
        weight_shape = (self.h, self.input_shape[1] + 1)  # Accounting for the bias term
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

        # Adding a dummy feature for bias
        X = util.add_dummy_feature(X)
        return np.dot(X, np.transpose(self.w))


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

    def back_propagation(self, f, d):
        """
        :param f:   Output of this layer during forward pass (N x C)
        :param d:   Gradients being passed back (N x C)
        :return: N x C numpy array of gradients
        """
        fd = f * d
        sum_fd = np.sum(fd, axis=1)
        return f * (d - sum_fd[:, None])


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

    def back_propagation(self, f, d):
        """
        :param f:   Output of this layer during forward pass (N x D)
        :param d:   Gradients being passed back (N x D)
        :return: N x D numpy array of gradients
        """
        return f * (1 - f) * d


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
        Y = np.copy(X)
        Y[X <= 0] = 0
        return Y

    def back_propagation(self, f, d):
        """
        :param f:   Output of this layer during forward pass (N x D)
        :param d:   Gradients being passed back (N x D)
        :return: N x D numpy array of gradients
        """
        return (f > 0) * d


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
        Y = np.copy(X)
        Y[X <= 0] *= self.alpha
        return Y

    def back_propagation(self, f, d):
        """
        :param f:   Output of this layer during forward pass (N x D)
        :param d:   Gradients being passed back (N x D)
        :return: N x D numpy array of gradients
        """
        new_d = np.ones(f.shape)
        new_d[f <= 0] = self.alpha
        return new_d * d


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

    def back_propagation(self, f, d):
        """
        :param f:   Output of this layer during forward pass (N x D)
        :param d:   Gradients being passed back (N x D)
        :return: N x D numpy array of gradients
        """
        return (1 - (f ** 2)) * d


class Loss:
    """
    Base class for Loss functions
    """

    def __init__(self):
        pass

    def get_loss_value(self, Y, O):
        pass

    def get_gradient(self, Y, O):
        pass


class CrossEntropy(Loss):
    """
    Cross-entropy loss
    """

    def get_loss_value(self, Y, O):
        """
        :param Y: Actual outputs (categorical) (N x C)
        :param O: Predicted outputs (N x C)
        :return: Column vector (N x 1) of loss values
        """
        return -1 * np.c_[np.sum(Y * np.log(O), axis=1)]

    def get_gradient(self, Y, O):
        """
        :param Y: Actual outputs (categorical) (N x C)
        :param O: Predicted outputs (N x C)
        :return:   Gradient numpy array (N x C)
        """
        return -Y / O


class Hinge(Loss):
    """
    Hinge loss
    """

    def get_loss_value(self, Y, O):
        """
        :param Y: Actual outputs (categorical) (N x C)
        :param O: Predicted outputs (N x C)
        :return: Column vector (N x 1) of loss values
        """
        # TODO Compute and return hinge loss value
        pass

    def get_gradient(self, Y, O):
        """
        :param Y: Actual outputs (categorical) (N x C)
        :param O: Predicted outputs (N x C)
        :return:   Gradient numpy array (N x C)
        """
        # TODO Compute and return the gradients for hinge loss
        pass
