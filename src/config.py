import argparse
from datetime import datetime
import os

import numpy as np
import pathlib


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config:

    def __init__(self):
        self.INFORMATION = self.information

        # input data
        args = self.parse_input
        self.DATA_DIR = args.data_dir
        self.DATA_SET_NAME = args.data_set_name
        self.LABELS_DIR = pathlib.Path(args.data_dir + args.labels_dir)
        self.LOGS_DIR = os.path.join(args.data_dir, args.logs_dir)
        self.CHECKPOINT_FILE = os.path.join(args.data_dir, args.checkpoint_file)
        self.MODEL_PATH = os.path.join(args.data_dir, args.model_path)
        self.PLOTS_DIR = os.path.join(args.data_dir, args.plots_dir)
        self.SHOW_MODEL_SUMMARY = bool(args.show_model_summary)
        self.GPU_0 = args.gpu_0
        self.GPU_1 = args.gpu_1
        self.RECONSTRUCTION_ON = bool(args.reconstruction_on)
        self.NUM_EPOCH = int(args.num_epoch)
        self.LEARNING_RATE = int(args.learning_rate)
        self.BATCH_SIZE = int(args.batch_size)
        self.IMG_HEIGHT = int(args.img_height)
        self.IMG_WIDTH = int(args.img_width)
        self.CHANNELS = int(args.channels)
        self.ROUTING_ALGO = str(args.routing_algo)
        self.DEBUG_MODE_ON = bool(args.debug_mode_on)
        self.NUM_OF_CAPSULE_LAYERS = int(args.number_of_caps_layers)

        image_count = len(list(self.LABELS_DIR.glob('*/*/*.png')) + list(self.LABELS_DIR.glob('*/*/*.jpg')))

        self.STEPS_PER_EPOCH = int(np.ceil(image_count / self.BATCH_SIZE))
        self.CLASS_NAMES = np.sort(np.array([item.name for item in self.LABELS_DIR.glob('*') if item.name != 'data']))

        self.NUM_ROUTING = 3

    def __str__(self):
        result = []
        for x in dir(self):
            if x.isupper():
                result += [x + ': ' + str(self.__dict__[x])]
        return '\n'.join(result)

    @property
    def parse_input(self):
        # parse the arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_dir', help='directory where the data is stored with the folder names = labels',
                            default='/media/DATA/lbieri/2019-11-08-_28px_28px_door_opener_3_labels_015_scalar_product_500epoch/')
        parser.add_argument('--data_set_name', help='name the data set which is used for plotting purposes',
                            default='Door Opener (3 Labels)')
        parser.add_argument('--labels_dir', help='the folder where the labels are stored',
                            default='labels/')
        parser.add_argument('--logs_dir', help='directory where the logs are being stored',
                            default='logs/' + datetime.now().strftime('%Y%m%d-%H%M%S'))
        parser.add_argument('--model_dir', help='directory where the logs are being stored',
                            default='model/')
        parser.add_argument('--checkpoint_file', help='define the folder where the checkpoints of the models are being saved',
                            default='ckpt/weights.ckpt')  # https://github.com/keras-team/keras/issues/10652 (no '{}' are allowed in the filename path
        parser.add_argument("--show_model_summary", type=str2bool, nargs='?',
                            const=True, default=True,
                            help='show the summary of the keras model to be trained')
        parser.add_argument('--gpu_0', help='Use the GPU:0 to process the model',
                            default=True)
        parser.add_argument('--gpu_1', help='Use the GPU:1 to process the model',
                            default=True)
        parser.add_argument('--model_path', help='The folder where the model shall be saved. (Do not enter the whole path to the folder.)',
                            default='model/model.h5')
        parser.add_argument("--reconstruction_on", type=str2bool, nargs='?',
                            const=True, default=True,
                            help='show the summary of the keras model to be trained')
        parser.add_argument('--num_epoch', help='The number of epochs used for training.',
                            default=500)
        parser.add_argument('--batch_size', help='The batch size used for training.',
                            default=32)
        parser.add_argument('--img_height', help='The height of the image in pixel.',
                            default=28)
        parser.add_argument('--img_width', help='The width of the image in pixel.',
                            default=28)
        parser.add_argument('--channels', help='The number of channels used in the picture. (if black and white => 1, if 3 colours => 3)',
                            default=1)
        parser.add_argument('--routing_algo',
                            help='The routing algorithm used in the capsule layer. (standard: \'scalar_product\', min-max: \'min_max\')',
                            default='scalar_product')
        parser.add_argument('--plots_dir', help='The folder where all the plots are being saved',
                            default='plots/')
        parser.add_argument('--learning_rate', help='Specify the learning rate for the training. (integer)',
                            default=0.001, type=int)
        parser.add_argument("--debug_mode_on", type=str2bool, nargs='?',
                            const=True, default=False,
                            help='turn the debug mode settings on (small dataset)')
        parser.add_argument('--number_of_caps_layers', help='Specify how many capsule layers shall be included. (Must be greater or equal to one.)',
                            type=lambda x: (x > 0) and x.is_integer(), default=1)

        args = parser.parse_args()
        return args

    @property
    def information(self):
        information = """
author:        Lukas Bieri
summary:       Training a neural network using a custom TensorFlow 2.0 implementation
               of Capsule Layers.
collaboration: ETH Zurich / Super Computing Systems AG
information:   - Currently it does not work on tensorflow-gpu (but runs on tensorflow)
               - run "python main.py --help" to customize this implementation of the capsule neural net
"""
        return information
