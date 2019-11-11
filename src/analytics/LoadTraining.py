import pandas as pd
import ast
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np


def fit_history_plot(history, path, data_set_name=None, algo_name=None):
    if ('output_1_accuracy' in history.keys()) and ('val_output_1_accuracy' in history.keys()):
        plt.figure()
        plt.plot(history['output_1_accuracy'])
        plt.plot(history['val_output_1_accuracy'])
        if (data_set_name is None) and (algo_name is None):
            title = 'Label Prediction Accuracy'
        else:
            title = str(data_set_name) + ' Accuracy ' + '(' + str(algo_name) + ')'
        plt.title(title)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.gca().set_yticklabels(['{:.0%}'.format(x) for x in plt.gca().get_yticks()])
        plt.xticks(np.linspace(start=0, stop=len(history['output_1_accuracy']), num=6, dtype=int))
        plt.grid(linestyle='-')
        plt.legend(['training', 'validation'], loc='upper left')
        plt.savefig(str(path) + datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + str(title).replace(' ', '_') + '.png')
    if ('accuracy' in history.keys()) and ('val_accuracy' in history.keys()):
        plt.figure()
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        if (data_set_name is None) and (algo_name is None):
            title = 'Label Prediction Accuracy'
        else:
            title = str(data_set_name) + ' Accuracy ' + '(' + str(algo_name) + ')'
        plt.title(title)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.gca().set_yticklabels(['{:.2%}'.format(x) for x in plt.gca().get_yticks()])
        plt.xticks(np.linspace(start=0, stop=len(history['accuracy']), num=6, dtype=int))
        plt.grid(linestyle='-')
        plt.legend(['training', 'validation'], loc='upper left')
        plt.savefig(str(path) + datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + str(title).replace(' ', '_') + '.png')

def plot_prediction(df, path):

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_title('Door Opener (Measurement Series Split) Percentage Of Correct Predictions (CNN)')
    chart = sns.heatmap(df, cmap='Blues', annot=True, vmin=0., vmax=1., cbar=True, ax=ax)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=30, horizontalalignment='right')
    # plt.show()
    plt.savefig(str(path) + datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + str('prediction') + '.png')


path = '/media/DATA/lbieri/2019-10-28-_28px_28px_door_opener_divided_by_location_010_scalar_recon/logs/20191028-174655/'
file = [
    'run-20191028-174655_train-tag-epoch_output_1_accuracy.csv',
    'run-20191028-174655_validation-tag-epoch_output_1_accuracy.csv'
]

plot_dir = path + '../../plots'

val_file_path = path + file[1]
tra_file_path = path + file[0]

val_df = pd.read_csv(val_file_path)
tra_df = pd.read_csv(tra_file_path)

if False:
    plt.figure()
    plt.plot(tra_df['Value'])
    plt.plot(val_df['Value'])
    plt.title('Door Opener Accuracy (Scalar Product Capsule)')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.gca().set_yticklabels(['{:.0%}'.format(x) for x in plt.gca().get_yticks()])
    plt.grid(linestyle='-')
    plt.legend(['training', 'validation'], loc='lower right')
    # plt.savefig(str(path) + datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + str('Door_Opener_Accuracy_Min-Max_Capsule') + '.png')

    plt.show()


path_history = '/media/DATA/lbieri/2019-10-28-mnist_009_cnn/plots/2019-10-30_11-59-47_fit_history.json'
with open(path_history) as json_file:
    history = ast.literal_eval(json_file.read())
playground = '/media/DATA/lbieri/2019-10-28-mnist_009_cnn/plots/'
fit_history_plot(history, playground, data_set_name='MNIST', algo_name='CNN')

path_prediction = '/media/DATA/lbieri/2019-10-28-mnist_009_cnn/plots/2019-10-30_11-59-52_prediction.pkl'
path_store = '/media/DATA/lbieri/2019-10-28-mnist_009_cnn/plots/'
with open(path_prediction, 'rb') as f:
    d = pickle.load(f)
plot_prediction(d, path_store)
