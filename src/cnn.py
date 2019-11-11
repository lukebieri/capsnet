import tensorflow as tf
import numpy as np


from analytics.analytics import fit_history_plot, save_model_summary, plot_prediction, evaluation_results
from Preprocessing.DataImport import DataImport
from Capsule.loss import margin_loss
from config import Config
from Capsule.save_squash import save_squash


config = Config()

# build model
inputs = tf.keras.Input(shape=(28, 28, 1,), batch_size=config.BATCH_SIZE, name='inputs')
conv1 = tf.keras.layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(inputs)
conv2 = tf.keras.layers.Conv2D(filters=256, kernel_size=9, strides=2, padding='valid')(conv1)
reshape1 = tf.keras.layers.Reshape(target_shape=[6 * 6 * 32, 8], name='primarycaps')(conv2)
squash = tf.keras.layers.Lambda(save_squash, name='squash')(reshape1)

reshape2 = tf.keras.layers.Reshape(target_shape=[6*6*32*8], name='reshape')(squash)
output_1 = tf.keras.layers.Dense(161, activation='relu')(reshape2)

predictions = tf.keras.layers.Dense(len(config.CLASS_NAMES), activation='softmax', name='predictions')(output_1)

model = tf.keras.Model(inputs=inputs, outputs=predictions)
model.summary()
save_model_summary(model, config.PLOTS_DIR)

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

# data preprocessing
ds = DataImport(config)
imgs, labels = ds.get_raw_data('train', num_show_imgs=None)
print('imgs.get_shape():   ', imgs.get_shape())
print('labels.get_shape(): ', labels.get_shape())

# train
history = model.fit(x=imgs, y=labels,
                    batch_size=config.BATCH_SIZE,
                    epochs=config.NUM_EPOCH,
                    validation_split=0.2)
fit_history_plot(history, config.PLOTS_DIR)

# evaluation
imgs_e, labels_e = ds.get_raw_data('test', num_show_imgs=None)
test_scores = model.evaluate(imgs_e, labels_e)
loss__ = 'Test loss:     {}'.format(test_scores[0])
accuracy__ = 'Test accuracy: {:,.6%}'.format(test_scores[1])
evaluation_results([loss__, accuracy__], config.PLOTS_DIR)

# prediction
labels_test_pred = model.predict(imgs_e)
pred = np.zeros_like(labels_test_pred)
for index, value in enumerate(np.argmax(labels_test_pred, axis=-1)):
    pred[index][value] = 1.0
labels_test_str = config.CLASS_NAMES[np.where(labels_e.numpy() == 1.0)[1]]
labels_test_pred_str = config.CLASS_NAMES[np.where(pred == 1.0)[1]]
plot_prediction(imgs_e, labels_test_str, labels_test_pred_str, config.PLOTS_DIR)
