import tensorflow as tf

from src.Capsule.save_squash import save_squash


def PrimaryCap(inputs, dim_vector, n_channels, kernel_size, strides, padding):
    """
    Apply Conv2D `n_channels` times and concatenate all capsules
    :param inputs: 4D tensor, shape=[None, width, height, channels]
    :param dim_vector: the dim of the output vector of capsule
    :param n_channels: the number of types of capsules
    :return: output tensor, shape=[None, num_capsule, dim_vector]
    """
    output = tf.keras.layers.Conv2D(filters=dim_vector*n_channels, kernel_size=kernel_size, strides=strides,
                                    padding=padding)(inputs)
    shape_tmp = output.get_shape()
    outputs = tf.keras.layers.Reshape(target_shape=[int(shape_tmp[1] * shape_tmp[2] * shape_tmp[3] / dim_vector),
                                                    dim_vector], name='primarycaps')(output)
    return tf.keras.layers.Lambda(save_squash, name='squash')(outputs)
