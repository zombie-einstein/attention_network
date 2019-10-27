from tensorflow.keras.layers import Layer, GRU, TimeDistributed, Bidirectional, Dense, Embedding
from tensorflow.keras import initializers, Sequential
from tensorflow.keras import Model
from attention_layer import AttLayer
from sklearn.preprocessing import normalize
import numpy as np


class HAN:
    def __init__(self, embedding_layer, sentence_width, model_width, input_shape, output_width):
        """
        initialize a new hierarchical network
        Args:
            embedding_layer (tf.keras.layers.Embedding): Pre-trained embedding layer
            sentence_width (int): Width of internal layers in word-level attention network
            model_width (int): Width of internal layers in sentence-level attention network
            input_shape (tuple): Input dimension shape tuple
            output_width (int): Output dimension (i.e. number of categories)
        """
        self.sentence_network = Sequential()
        self.sentence_network.add(embedding_layer)
        self.sentence_network.add(Bidirectional(GRU(sentence_width, return_sequences=True)))
        self.sentence_network.add(AttLayer(sentence_width))

        self.model = Sequential()
        self.model.add(TimeDistributed(self.sentence_network, input_shape=input_shape))
        self.model.add(Bidirectional(GRU(model_width, return_sequences=True)))
        self.model.add(AttLayer(model_width))
        self.model.add(Dense(output_width, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])

    def summary(self):
        """
        Print summary of network
        """
        self.sentence_network.summary()
        self.model.summary()

    def fit(self, x_train, y_train, x_val, y_val, epochs, batch_size):
        """
        Train the network
        Args:
            x_train (np.array): Training inputs
            y_train (np.array): Training target labels
            x_val (np.array): Validation inputs
            y_val (np.array): Validation target labels
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
        """
        self.model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size)
        self.hidden_word_output = Model(self.sentence_network.input, self.sentence_network.layers[1].output)
        self.word_ctx_0 = self.sentence_network.layers[-1].get_weights()[0]
        self.word_ctx_1 = self.sentence_network.layers[-1].get_weights()[1]
        self.word_ctx_2 = self.sentence_network.layers[-1].get_weights()[2]
        self.hidden_sent_output = Model(self.model.input, self.model.layers[-3].output)
        self.sent_ctx_0 = self.model.layers[-2].get_weights()[0]
        self.sent_ctx_1 = self.model.layers[-2].get_weights()[1]
        self.Sent_ctx_2 = self.model.layers[-2].get_weights()[2]

    def predict(self, x):
        """
        Predict category of sample(s)
        Args:
            x (np.array): Input samples(s)

        Returns:
            (np.array): Category predictions array
        """
        return self.model.predict(x)

    def attention_matrix(self, x):
        """
        Get per-word attention matrix
        Args:
            x (np.array): Input sample

        Returns:
            (np.array): Per word attention values, normalized over all words
        """
        word_att = self.hidden_word_output.predict(x)
        u_watt = np.exp(np.dot(np.tanh(np.dot(word_att, self.word_ctx_0) + self.word_ctx_1), self.word_ctx_2)[:, :, 0])
        u_watt = normalize(u_watt, axis=1, norm='l2')

        sent_att = self.hidden_sent_output.predict(np.expand_dims(x, axis=0))
        u_satt = np.exp(np.dot(np.tanh(np.dot(sent_att, self.sent_ctx_0)+self.sent_ctx_1), self.Sent_ctx_2)[0, :, 0])
        u_satt = normalize(u_satt, axis=1, norm='l2')

        return u_satt*u_watt
