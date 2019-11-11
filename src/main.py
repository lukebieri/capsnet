from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from analytics.analytics import dataset_hist, fit_history_plot, evaluation_results, save_model_summary, plot_prediction, store_config
from Capsule.CapsNet import CapsNet
from config import Config
from Preprocessing.DataImport import DataImport
from Capsule.loss import margin_loss
from model import get_model


def model_compile(model, config):
    # compile model
    if config.RECONSTRUCTION_ON:
        model.compile(optimizer='adam',
                      loss=[tf.keras.losses.categorical_crossentropy, 'mse'],
                      loss_weights=[1., .0005],
                      metrics={'output_1': ['accuracy'], 'output_2': ['accuracy']})
    else:
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.categorical_crossentropy,  # margin_loss
                      metrics={'output_1': ['accuracy']})


def main():

    # parse config
    config = Config()
    store_config('capsule network', config, config.PLOTS_DIR)

    # tf.debugging.set_log_device_placement(True)

    # # allowed GPU to use
    # if not (config.GPU_0 and config.GPU_1):
    #     if config.GPU_0:
    #         os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    #     elif config.GPU_1:
    #         os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    #     assert config.GPU_0 or config.GPU_1, 'No GPU is set to process the ML model.'
    #

    # tf.keras.utils.multi_gpu_model(model, 2, cpu_merge=True, cpu_relocation=False)
    # input_shape = (config.IMG_WIDTH, config.IMG_HEIGHT, config.CHANNELS,)
    # model = get_model(config)
    model = CapsNet(config)
    model_compile(model, config)


    # import data from the config.LABELS_DIR directory (issues with tensorflow.data.Dataset)
    ds = DataImport(config)
    imgs, labels = ds.get_raw_data('train', num_show_imgs=None)
    validation_split = 0.2
    imgs_train = imgs[:int((1 - validation_split) * len(imgs))]
    labels_train = labels[:int((1 - validation_split) * len(labels))]
    imgs_validation = imgs[int(-validation_split * len(imgs)):]
    labels_validation = labels[int(-validation_split * len(labels)):]
    imgs_test, labels_test = ds.get_raw_data('test', num_show_imgs=None)

    # debug settings
    if config.DEBUG_MODE_ON:
        num_samples = 1280
        imgs_train = imgs_train[:num_samples]
        labels_train = labels_train[:num_samples]
        print('test size:        ', len(imgs_train), len(labels_train))
        imgs_validation = imgs_validation[:int(num_samples / 5)]
        labels_validation = labels_validation[:int(num_samples / 5)]
        print('validation size:  ', len(imgs_validation), len(labels_validation))
        imgs_test = imgs_test[:int(num_samples / 10)]
        labels_test = labels_test[:int(num_samples / 10)]
        print('test size:        ', len(imgs_test), len(labels_test))
        config.NUM_EPOCH = 3
        print('number of epochs: ', config.NUM_EPOCH)

    # data analysis
    dataset_hist(config.CLASS_NAMES[np.where(labels_train.numpy() == 1)[1]], 'Training', config.PLOTS_DIR)
    dataset_hist(config.CLASS_NAMES[np.where(labels_validation.numpy() == 1)[1]], 'Validation', config.PLOTS_DIR)
    dataset_hist(config.CLASS_NAMES[np.where(labels_test.numpy() == 1)[1]], 'Testing', config.PLOTS_DIR)

    # callbacks
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=config.LOGS_DIR)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(config.CHECKPOINT_FILE,
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    verbose=1)
    # lr_decay = tf.keras.callbacks.LearningRateScheduler(schedule=lambda epoch: config.LEARNING_RATE * np.exp(-epoch / float(config.NUM_EPOCH)))

    if config.RECONSTRUCTION_ON:

        # train
        history = model.fit(x={'input_1': imgs_train, "input_2": labels_train},
                            y=([labels_train, imgs_train]),
                            batch_size=config.BATCH_SIZE,
                            epochs=config.NUM_EPOCH,
                            validation_data=({'input_1': imgs_validation, "input_2": labels_validation},
                                             ([labels_validation, imgs_validation])),
                            shuffle=True,
                            callbacks=[tensorboard,
                                       checkpoint,
                                       #lr_decay
                                       ]
                            )
        fit_history_plot(history, config.PLOTS_DIR, data_set_name=config.DATA_SET_NAME,
                         algo_name=config.ROUTING_ALGO.replace('_', ' ').title())

        # evaluate
        eval_output = model.evaluate(x={'input_1': imgs_test, 'input_2': labels_test},
                                     y=([labels_test, imgs_test]),
                                     batch_size=config.BATCH_SIZE
                                     )
        [loss, y_prob_loss, recon_loss, y_prob_accuracy, recon_accuracy] = eval_output
        loss__ = 'loss:            {:.5}'.format(loss)
        y_prob_loss = 'y_prob_loss:     {:.5}'.format(y_prob_loss)
        recon_loss__ = 'recon_loss:      {:.5}'.format(recon_loss)
        y_prob_accuracy__ = 'y_prob_accuracy: {:.3%}'.format(y_prob_accuracy)
        recon_accuracy__ = 'recon_accuracy:  {:.3%}'.format(recon_accuracy)
        evaluation_results([loss__, y_prob_loss, recon_loss__, y_prob_accuracy__, recon_accuracy__], config.PLOTS_DIR)

        # prediction
        [labels_test_pred, imgs_test_recon] = model.predict(x={'input_1': imgs_test, 'input_2': labels_test})
        pred = np.zeros_like(labels_test_pred)
        for index, value in enumerate(np.argmax(labels_test_pred, axis=-1)):
            pred[index][value] = 1.0
        labels_test_str = config.CLASS_NAMES[np.where(labels_test.numpy() == 1.0)[1]]
        labels_test_pred_str = config.CLASS_NAMES[np.where(pred == 1.0)[1]]
        plot_prediction(imgs_test, labels_test_str, labels_test_pred_str, config.PLOTS_DIR)


    elif not config.RECONSTRUCTION_ON:

        # training
        history = model.fit(x=imgs_train,
                            y=labels_train,
                            batch_size=config.BATCH_SIZE,
                            epochs=config.NUM_EPOCH,
                            validation_data=(imgs_validation,
                                             labels_validation),
                            shuffle=True,
                            callbacks=[tensorboard,
                                       checkpoint,
                                       # lr_decay
                                       ]
                            )
        fit_history_plot(history, config.PLOTS_DIR)

        # evaluate
        [loss_, accuracy_] = model.evaluate(x=imgs_test, y=labels_test, batch_size=config.BATCH_SIZE, )
        loss__ = 'loss:     {:.5}'.format(loss_)
        accuracy__ = 'accuracy: {:.3%}'.format(accuracy_)
        evaluation_results([loss__, accuracy__], config.PLOTS_DIR)

        # prediction
        labels_test_pred = model.predict(imgs_test)
        pred = np.zeros_like(labels_test_pred)
        for index, value in enumerate(np.argmax(labels_test_pred, axis=-1)):
            pred[index][value] = 1.0
        labels_test_str = config.CLASS_NAMES[np.where(labels_test.numpy() == 1.0)[1]]
        labels_test_pred_str = config.CLASS_NAMES[np.where(pred == 1.0)[1]]
        plot_prediction(imgs_test, labels_test_str, labels_test_pred_str, config.PLOTS_DIR)

    # model summary
    save_model_summary(model, config.PLOTS_DIR)
    if config.SHOW_MODEL_SUMMARY:
        model.summary()
        try:
            tf.keras.utils.vis_utils.plot_model(model, to_file='model.png', show_shapes=True)
        except Exception as e:
            print('No fancy plot {}'.format(e))

if __name__ == "__main__":
    main()
