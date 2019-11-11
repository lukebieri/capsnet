import tensorflow as tf


def test(model, config, ds=None, imgs=None, labels=None):

    # load the model
    if model is None:
        model = tf.keras.models.load_model(config.MODEL_PATH)

    # evaluate the model with the testing data
    if ds is not None:
        loss, accuracy = model.evaluate(ds)
    elif (imgs is not None) and (labels is not None):
        space_holder = tf.keras.backend.zeros_like(labels)
        [loss, y_prob_loss, out_recon_loss, y_prob_accuracy, out_recon_accuracy, out_recon_mse] =\
            model.evaluate(x={'input_1': imgs, "input_2": space_holder},
                           y={'y_prob': labels, 'out_recon': imgs})
    print([loss, y_prob_loss, out_recon_loss, y_prob_accuracy, out_recon_accuracy, out_recon_mse])
    print('accuracy: ', y_prob_accuracy * 100., '%')
    return loss, accuracy

# [0.851703330039978, 0.8515919,       0.219791,         0.107883334]
# ['loss',            'out_caps_loss', 'out_recon_loss', 'out_caps_accuracy']