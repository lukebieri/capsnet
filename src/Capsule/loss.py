import tensorflow as tf


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """

    m_plus = 0.9
    m_minus = 0.1
    lamb = 0.5

    L = y_true * tf.keras.backend.square(tf.keras.backend.maximum(0., m_plus - y_pred)) +\
        lamb * (1. - y_true) * tf.keras.backend.square(tf.keras.backend.minimum(0., y_pred - m_minus))
    return tf.keras.backend.mean(tf.keras.backend.sum(L, axis=1))


def reconstruction_loss(img_original, img_reconstructed):
    assert img_original.get_shape() == img_original.get_shape(), 'The original image and the reconstructed image must' \
                                                                 'have the same shape.'
    img_dim = img_original.get_shape()[-1] * img_original.get_shape()[-1]
    return tf.keras.backend.mean(tf.keras.backend.reshape(tf.keras.backend.square(img_original-img_reconstructed),
                                                          [-1, img_dim]))
