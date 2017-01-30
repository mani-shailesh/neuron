import numpy as np


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
        self.Y = None
        self.X = None

    def get_output_shape(self):
        """
        Return the output shape of this layer
        :return: tuple
        """
        pass

    def forward_pass(self, X):
        """
        Perform forward pass on the input
        :param X: input numpy array (N x D)
        :return: numpy array - (N x H)
        """
        pass

    def back_propagation(self, d, lr):
        """
        Computes gradients and updates the weights of this layer
        :param d:   Gradients being passed back (N x H)
        :param lr:  Learning rate
        :return: N x D numpy array of gradients
        """
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

    def __init__(self, num_units, weight_decay=0.0, *args, **kwargs):
        """
        Initialization of fully connected layer
        :param num_units: Number of units in this layer
        :param weight_decay: Weight decay for L2 regularization
        """
        Layer.__init__(self, *args, **kwargs)
        self.h = num_units
        self.weight_decay = weight_decay
        weight_shape = (self.h, self.input_shape[1])
        self.w = np.random.standard_normal(weight_shape) * np.sqrt(2.0 / self.input_shape[1])
        self.b = np.zeros((self.h, 1))  # bias weights

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
        :param X: input numpy array (N x D)
        :return: numpy array - (N x H)
        """
        self.X = np.copy(X)
        return np.dot(X, np.transpose(self.w)) + np.transpose(self.b)

    def back_propagation(self, d, lr):
        """
        Computes gradients and updates the weights of this layer
        :param d:   Gradients being passed back (N x H)
        :param lr:  Learning rate
        :return: N x D numpy array of gradients
        """
        new_d = np.dot(d, self.w)
        d_w = np.dot(np.transpose(d), self.X) / self.input_shape[0] \
              + self.weight_decay * self.w
        d_b = np.c_[np.sum(d, axis=0)] / self.input_shape[0]
        self.w -= lr * d_w
        self.b -= lr * d_b
        return new_d


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
        self.Y = np.exp(X) / row_sum[:, None]
        return self.Y

    def back_propagation(self, d, lr):
        """
        :param d:   Gradients being passed back (N x C)
        :param lr:  Learning rate
        :return: N x C numpy array of gradients
        """
        fd = self.Y * d
        sum_fd = np.sum(fd, axis=1)
        return self.Y * (d - sum_fd[:, None])


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
        self.Y = 1 / (1 + np.exp(-1 * X))
        return self.Y

    def back_propagation(self, d, lr):
        """
        :param d:   Gradients being passed back (N x D)
        :param lr:  Learning rate
        :return: N x D numpy array of gradients
        """
        return self.Y * (1 - self.Y) * d


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
        self.Y = np.copy(X)
        self.Y[X <= 0] = 0
        return self.Y

    def back_propagation(self, d, lr):
        """
        :param d:   Gradients being passed back (N x D)
        :param lr:  Learning rate
        :return: N x D numpy array of gradients
        """
        return (self.Y > 0) * d


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
        self.Y = np.copy(X)
        self.Y[X <= 0] *= self.alpha
        return self.Y

    def back_propagation(self, d, lr):
        """
        :param d:   Gradients being passed back (N x D)
        :param lr:  Learning rate
        :return: N x D numpy array of gradients
        """
        new_d = np.ones(self.Y.shape)
        new_d[self.Y <= 0] = self.alpha
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
        self.Y = np.tanh(X)
        return self.Y

    def back_propagation(self, d, lr):
        """
        :param d:   Gradients being passed back (N x D)
        :param lr:  Learning rate
        :return: N x D numpy array of gradients
        """
        return (1 - (self.Y ** 2)) * d


class Loss:
    """
    Base class for Loss functions
    """

    def __init__(self):
        pass

    def get_loss_value(self, Y, O):
        """
        :param Y: Actual outputs (categorical) (N x C)
        :param O: Predicted outputs (N x C)
        :return: Column vector (N x 1) of loss values
        """
        pass

    def get_gradient(self, Y, O):
        """
        :param Y: Actual outputs (categorical) (N x C)
        :param O: Predicted outputs (N x C)
        :return:   Gradient numpy array (N x C)
        """
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
        oyi = O[range(O.shape[0]), np.argmax(Y, axis=1)]
        l = (O - oyi[:, None]) + 1
        l[l < 0] = 0
        return np.c_[np.sum(l, axis=1) - 1]

    def get_gradient(self, Y, O):
        """
        :param Y: Actual outputs (categorical) (N x C)
        :param O: Predicted outputs (N x C)
        :return:   Gradient numpy array (N x C)
        """
        # Compute and return the gradients for hinge loss
        oy = O[range(O.shape[0]), np.argmax(Y, axis=1)]
        l = (O - oy[:, None]) + 1
        grad = np.zeros(O.shape)
        grad[l > 0] = 1
        grad[range(O.shape[0]), np.argmax(Y, axis=1)] = -1 * (np.sum(grad, axis=1) - 1)
        return grad
