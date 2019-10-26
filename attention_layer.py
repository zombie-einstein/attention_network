import tensorflow.keras.backend as K
from tensorflow.keras import Variable
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers


class AttLayer(Layer):
    """
    Attention layer for use with Keras

    adapted from https://github.com/richliao/textClassifier for tensorflow 2.0
    """
    def __init__(self, attention_dim):
        """
        Initialize a new attention layer
        Args:
            attention_dim (int): Internal dimension of this layer
        """
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        """
        Initialize the internal matrices of the layer given input shape
        Args:
            input_shape (tuple): Shape of input layer (3D)
        """
        assert len(input_shape) == 3
        self.W = Variable(self.init((input_shape[-1], self.attention_dim)), trainable=True)
        self.b = Variable(self.init((self.attention_dim,)), trainable=True)
        self.u = Variable(self.init((self.attention_dim, 1)), trainable=True)
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        """
        Calculate the output of this layer
        Args:
            x: Input tensor
            mask (default None):

        Returns:
            output tensor
        """
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)
        ait = K.exp(ait)

        if mask is not None:
            ait *= K.cast(mask, K.floatx())

        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]
