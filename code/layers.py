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
        self.n = num_units
        weight_shape = (self.n, self.input_shape[1] + 1)
        self.w = np.zeros(weight_shape)


class Softmax(Layer):
    """
    Softmax activation layer
    """
    pass


class ReLU(Layer):
    """
    ReLU activation layer
    """
    pass
