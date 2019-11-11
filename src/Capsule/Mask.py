import tensorflow as tf


class Mask(tf.keras.layers.Layer):
    def call(self, inputs, training=None, **kwargs):
        # digitcaps.shape = [batch_size, num_labels]
        # y.shape = [batch_size, num_labels]
        [digitcaps, y] = inputs



        # during non-training predictions we simply forward the layer
        if not training:
            return digitcaps

        # for training we put all weights to zero except the ones from the capsules that are correct
        # mask.shape = [batch_size, dim_vector]
        y_tile = tf.tile(tf.expand_dims(y, -1), [1, 1, 16], name='y_tile')
        mask = tf.math.multiply(tf.dtypes.cast(y_tile, digitcaps.dtype), digitcaps, name='mask')
        return mask

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list), 'as an input we want a list of [(data_from_previous_layer), (y)]'
        [output, y] = input_shape
        assert isinstance(output, tuple)
        return tuple([output])
