import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm


class DataImport:

    def __init__(self, config):
        self.config = config
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        if 'mnist' in str(self.config.LABELS_DIR):
            self.files_training = [str(i) for i in Path(self.config.LABELS_DIR).rglob('*/training/*.png')]
            self.files_testing = [str(i) for i in Path(self.config.LABELS_DIR).rglob('*/testing/*.png')]
        elif 'door_opener_divided_by_location' in str(self.config.LABELS_DIR):
            self.files_training = [str(i) for i in Path(self.config.LABELS_DIR).rglob('*/training/*/*.jpg')]
            self.files_testing = [str(i) for i in Path(self.config.LABELS_DIR).rglob('*/testing/*/*.jpg')]
        elif 'door' in str(self.config.LABELS_DIR):
            files = [str(i) for i in Path(self.config.LABELS_DIR).rglob('*/*/*.jpg')]
            random.shuffle(files)
            num_train = int(0.8 * len(files))
            self.files_training = files[:num_train]
            self.files_testing = files[num_train:]
        else:
            assert False, 'Could not read any files'
        random.shuffle(self.files_training)
        random.shuffle(self.files_testing)

    def get_label(self, file_path):
        parts = tf.strings.split(file_path, '/')
        return parts[6] == self.config.CLASS_NAMES

    def decode_img(self, img):
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return tf.image.resize(img, [self.config.IMG_WIDTH, self.config.IMG_HEIGHT])

    def process_path(self, file_path):
        label = self.get_label(file_path)
        label = tf.dtypes.cast(label, tf.dtypes.float32)
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img, label

    def prepare_for_training(self, ds, cache=True, shuffle_buffer_size=1000):
        # This is a small dataset, only load it once, and keep it in memory.
        # use `.cache(filename)` to cache preprocessing work for datasets that don't
        # fit in memory.
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()

        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

        # Repeat forever
        ds = ds.repeat()

        ds = ds.batch(self.config.BATCH_SIZE)

        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)

        return ds

    def show_batch(self, image_batch, label_batch, show_plot=True):
        if np.shape(image_batch)[-1] == 1:
            image_batch = image_batch.reshape((-1, self.config.IMG_WIDTH, self.config.IMG_HEIGHT))
        plt.figure(figsize=(10, 10))
        for n in range(25):
            ax = plt.subplot(5, 5, n + 1)
            plt.imshow(image_batch[n], cmap='binary')
            plt.title(self.config.CLASS_NAMES[label_batch[n] == 1][0].title())
            plt.axis('off')
        if show_plot:
            plt.show()

    def get_raw_data(self, test_or_train, num_show_imgs=None):
        # input validation
        assert (test_or_train is 'test') or (test_or_train is 'train'),\
            'The variable \'test_or_train\' must be set to either \'test\' or \'train\'.'

        # function constants
        file_img = Path(str(self.config.LABELS_DIR) +
                        str(Path('/data/' + test_or_train + '-img-' + str(self.config.IMG_WIDTH) + '-' + str(
                            self.config.IMG_HEIGHT) + '.npy')))
        file_lab = Path(str(self.config.LABELS_DIR) +
                        str(Path('/data/' + test_or_train + '-lab-' + str(self.config.IMG_WIDTH) + '-' + str(
                            self.config.IMG_HEIGHT) + '.npy')))
        if not os.path.exists(str(self.config.LABELS_DIR) + '/data/'):
            os.mkdir(str(self.config.LABELS_DIR) + '/data/')
        if test_or_train is 'test':
            files = self.files_testing
        elif test_or_train is 'train':
            files = self.files_training

        if file_img.exists() and file_lab.exists():
            imgs = np.load(file_img)
            labels = np.load(file_lab)
        else:
            img, y = self.process_path(files[0])
            imgs = tf.keras.backend.expand_dims(img, 0)
            labels = tf.keras.backend.expand_dims(y, 0)
            print('parsing image data to load into the model')
            for file_path in tqdm(files[1:]):
                img, y = self.process_path(file_path)
                imgs = tf.concat([imgs, tf.keras.backend.expand_dims(img, 0)], 0)
                labels = tf.concat([labels, tf.keras.backend.expand_dims(y, 0)], 0)

            np.save(file_img, imgs)
            np.save(file_lab, labels)
        if num_show_imgs is not None:
            if tf.is_tensor(imgs):
                imgs = imgs.numpy()
            if tf.is_tensor(labels):
                labels = labels.numpy()
            plt.figure(figsize=(10, 10))
            assert isinstance(num_show_imgs, int)
            for index, n in enumerate([random.randint(0, len(imgs)-1) for _ in range(num_show_imgs)]):
                if np.shape(imgs[n])[-1] == 1:
                    image = imgs[n].reshape((self.config.IMG_WIDTH, self.config.IMG_HEIGHT))
                ax = plt.subplot(5, 5, index + 1)
                plt.imshow(image, cmap='binary')
                plt.title(self.config.CLASS_NAMES[labels[n] == 1][0].title())
                plt.axis('off')
            plt.show()

        return tf.constant(imgs), tf.constant(labels)

    def get_training_iter(self, show_plot=False, show_data=False):
        def generator():
            for file_path in self.files_training:
                img, y = self.process_path(file_path)
                yield {'input_1': img, "input_2": y}, {'out_caps': y, 'out_recon': img}

        # https://github.com/tensorflow/tensorflow/issues/24570 from_generator may create issues
        labeled_ds = tf.data.Dataset.from_generator(generator,
                                                    output_types=(
                                                            {'input_1': tf.float32, 'input_2': tf.float32},
                                                            {'out_caps': tf.float32, 'out_recon': tf.float32}))

        # file_img = Path(self.config.LABELS_DIR +
        #                 Path('data/training-img-' + str(self.config.IMG_WIDTH) + '-' +str(self.config.IMG_HEIGHT) +'.npy'))
        # file_lab = Path(self.config.LABELS_DIR +
        #                 Path('data/training-lab-' + str(self.config.IMG_WIDTH) + '-' + str(
        #                     self.config.IMG_HEIGHT) + '.npy'))
        # if file_img.exists() and file_lab.exists:
        #     img = np.load(file_img)
        # else:
        #     img, y = self.process_path(self.files_training[0])
        #     imgs = tf.keras.backend.expand_dims(img, 0)
        #     labels = tf.keras.backend.expand_dims(y, 0)
        #     for file_path in self.files_training[1:]:
        #         img, y = self.process_path(file_path)
        #         imgs = tf.concat([imgs, tf.keras.backend.expand_dims(img, 0)], 0)
        #         labels = tf.concat([labels, tf.keras.backend.expand_dims(y, 0)], 0)
        #
        # labeled_ds = tf.data.Dataset.from_tensor_slices(({'input_1': imgs, "input_2": labels},
        #                                                  {'out_caps': labels, 'out_recon': imgs}))

        if show_data:
            # show training data which has been imported
            for input, output in labeled_ds.take(1):
                print('image shape: ', input['input_1'].numpy().shape)
                print('Label: ', input['input_2'].numpy())

        train_ds = self.prepare_for_training(labeled_ds)

        # show a sample of the batch
        if show_plot:
            input, output = next(iter(train_ds))
            self.show_batch(input['input_1'].numpy(), input['input_2'].numpy(), show_plot=False)

        return train_ds

    def get_testing_iter(self, show_plot=False, show_data=False):
        def generator():
            for file_path in self.files_testing:
                img, y = self.process_path(file_path)
                yield {'input_1': img, "input_2": y}, {'out_caps': y, 'out_recon': img}

        labeled_ds = tf.data.Dataset.from_generator(generator,
                                                    output_types=(
                                                            {'input_1': tf.float32, 'input_2': tf.float32},
                                                            {'out_caps': tf.float32, 'out_recon': tf.float32}))

        if show_data:
            # show testing data which has been imported
            for input, output in labeled_ds.take(1):
                print('image shape: ', input['input_1'].numpy().shape)
                print('Label: ', input['input_2'].numpy())

        test_ds = self.prepare_for_training(labeled_ds)

        # show a sample of the batch
        if show_plot:
            input, output = next(iter(test_ds))
            self.show_batch(input['input_1'].numpy(), input['input_2'].numpy(), show_plot=False)

        return test_ds
