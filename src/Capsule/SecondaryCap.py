import tensorflow as tf


class SecondaryCap(tf.keras.layers.Layer):

    def call(self, inputs, **kwargs):
        # input.shape = [batch_size, 10, 16]
        shape = inputs.get_shape()
        return tf.reshape(inputs, [-1, shape[-1] * shape[-2]], name='secondary_cap')

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, tuple), 'as an input we want a list of [(data_from_previous_layer), (y)]'
        return tuple([input_shape[0][0], input_shape[0][-1] * input_shape[0][-2]])