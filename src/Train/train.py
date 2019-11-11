from datetime import datetime
import tensorflow as tf
import numpy as np


def train(model, config, ds=None, train_x=None, train_y=None):
    """
    Training a CapsuleNet
    :param config.STEPS_PER_EPOCH: number of steps per epoch
    :param model: the CapsuleNet model
    :param ds: a tf.data data set which returns (input, targets)`
    :return: The trained model
    """

    # callbacks
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=config.LOGS_DIR)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(config.CHECKPOINT_FILE,
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    verbose=1)
    lr_decay = tf.keras.callbacks.LearningRateScheduler(schedule=lambda epoch: 0.001 * np.exp(-epoch / 10.))

    # # see https://github.com/tensorflow/tensorflow/issues/32912
    # # currently model.fit has an issue and cannot be used with a tf.data.Dataset
    # model.fit(ds,
    #           steps_per_epoch=config.STEPS_PER_EPOCH,
    #           callbacks=[tensorboard])

    # tf.data.Datastore has issues
    if config.RECONSTRUCTION_ON:
        if ds is not None:
            model.fit_generator(ds,
                                steps_per_epoch=config.STEPS_PER_EPOCH,
                                callbacks=[tensorboard,
                                           checkpoint,
                                           lr_decay])
        elif (train_x is not None) and (train_y is not None):
            model.fit(x={'input_1': train_x, "input_2": train_y},
                      y={'y_prob': train_y, 'out_recon': train_x},
                      batch_size=config.BATCH_SIZE,
                      epochs=config.NUM_EPOCH,
                      shuffle=True,
                      validation_split=0.2,
                      steps_per_epoch=config.STEPS_PER_EPOCH,
                      callbacks=[tensorboard,
                                 checkpoint])
        else:
            print('Could not train')
    else:
        if (train_x is not None) and (train_y is not None):

            model.fit(x=train_x, y=train_y,
                      batch_size=config.BATCH_SIZE,
                      epochs=config.NUM_EPOCH,
                      shuffle=True,
                      validation_split=0.2,
                      steps_per_epoch=config.STEPS_PER_EPOCH,
                      callbacks=[tensorboard,
                                 checkpoint])
        else:
            print('Could not train')



    # tf.train.Checkpoint.save(config.CHECKPOINT_FILE)
    # model.save(config.MODEL_PATH)
    return model
