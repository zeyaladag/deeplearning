import tensorflow as tf

class myRNNCell(tf.keras.layers.Layer):
    def __init__(self, rnn_units, input_dim, output_dim):
        super(myRNNCell, self).__init__()
        # initialize weights and biases
        self.Wh = self.add_weight([rnn_units, input_dim])
        self.Wx = self.add_weight([rnn_units, rnn_units])
        self.b = self.add_weight([output_dim, rnn_units])

        # initialize hidden state to zeros
        self.h = tf.zeros([rnn_units,1])

    def call(self, x):
        self.h =tf.math.tanh(self.W_hh * self.h + self.W_xh * x + x) # update hidden state
        output = self.W_hy * self.h # compute output
        return output, self.h # return the current output and hidden state