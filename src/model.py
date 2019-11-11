import tensorflow as tf

# own classes
from Capsule.CapsuleLayer import CapsuleLayer
from Capsule.Length import Length
from Capsule.Mask import Mask
from Capsule.save_squash import save_squash


def get_model(config, input_shape=(28, 28, 1)):
    conv1_strides = 1
    conv1_kernel_size = 9
    conv1_filters = 256

    conv1_output_width = int(((config.IMG_WIDTH - conv1_kernel_size) / conv1_strides) + 1)
    conv1_output_height = int(((config.IMG_HEIGHT - conv1_kernel_size) / conv1_strides) + 1)

    conv2_strides = 2
    conv2_kernel_size = 9

    conv2_output_width = int(((conv1_output_width - conv2_kernel_size) / conv2_strides) + 1)
    conv2_output_height = int(((conv1_output_height - conv2_kernel_size) / conv2_strides) + 1)

    primary_capsule_input_dim = 8
    primary_capsule_map = 32
    capsule_output_dim = 16

    inputs = tf.keras.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = tf.keras.layers.Conv2D(filters=conv1_filters,
                                   kernel_size=conv1_kernel_size,
                                   strides=conv1_strides,
                                   padding='valid',
                                   activation='relu',
                                   name='conv1')(inputs)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_vector]
    conv2 = tf.keras.layers.Conv2D(filters=primary_capsule_input_dim * primary_capsule_map,
                                   kernel_size=conv2_kernel_size,
                                   strides=conv2_strides,
                                   padding='valid')(conv1)
    reshape1 = tf.keras.layers.Reshape(
        target_shape=[conv2_output_width * conv2_output_height * primary_capsule_map, primary_capsule_input_dim, ],
        name='primarycaps')(conv2)
    squash = tf.keras.layers.Lambda(save_squash, name='squash')(reshape1)

    # Layer 3: Capsule layer. Routing algorithm works here.

    if config.NUM_OF_CAPSULE_LAYERS > 1:
        caps_1 = CapsuleLayer(num_capsule=len(config.CLASS_NAMES),
                              dim_vector=capsule_output_dim,
                              num_routing=config.NUM_ROUTING,
                              routing_algo=config.ROUTING_ALGO,
                              name='caps_1')(squash)
    else:
        caps_1 = squash
    caps_out = CapsuleLayer(num_capsule=len(config.CLASS_NAMES),
                            dim_vector=capsule_output_dim,
                            num_routing=config.NUM_ROUTING,
                            routing_algo=config.ROUTING_ALGO,
                            name='caps_out')(caps_1)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    y_prob = Length(name='output_1')(caps_out)

    return tf.keras.Model(inputs=inputs, outputs=y_prob)
