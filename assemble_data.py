import os
import pandas as pd
import scipy.io.wavfile as sciw
from feature_extraction import create_feature_vector
import numpy as np
from fileutils import TRAIN_DATA_DIR, PROJECT_DIR
import csv

TRAIN_FEATURES_FILE = os.path.join(PROJECT_DIR, 'train_features.csv')
TRAIN_LABELS_FILE = os.path.join(PROJECT_DIR, 'train_labels.csv')

def prepare_training_data():
    """
    This function walks through directories in train, and reads touch.csv and audio.wav files.
    It invokes create_feature_vector function to extract features, extracts labels and writes them to
    corresponding csv files.
    :return:
    """

    top_level_train_dirs = os.listdir(TRAIN_DATA_DIR)
    top_level_train_dirs = [x for x in top_level_train_dirs if not x.startswith('.')]

    # Remove prep data files if exist
    try:
        os.remove(TRAIN_FEATURES_FILE)
        os.remove(TRAIN_LABELS_FILE)
    except OSError:
        pass

    # Write header to the csv file that should contain feature matrix
    with open(TRAIN_FEATURES_FILE, 'wb') as write_header:
        writer = csv.writer(write_header)
        writer.writerow(['x', 'y', 'major', 'minor', 'pressure', 'orientation', 'signal energy',
                         'signal energy entropy', 'spectral centroid', 'spectral spread', 'spectral entropy',
                         'spectral roll off', 'mode'])

    # Walk through directories in train directory and extract features from files,
    # along with label from directory names

    for dir_name in top_level_train_dirs:
        mode = dir_name.split('-')[0]
        timestamped_dirs = os.listdir(os.path.join(TRAIN_DATA_DIR, dir_name))
        timestamped_dirs = [x for x in timestamped_dirs if not x.startswith('.')]
        for folder_name in timestamped_dirs:
            split_details = folder_name.split('-')
            label = split_details[1]
            # timestamp = split_details[0]
            audio_file = os.path.join(TRAIN_DATA_DIR, dir_name, folder_name, 'audio.wav')
            touch_file = os.path.join(TRAIN_DATA_DIR, dir_name, folder_name, 'touch.csv')
            touch_features = pd.read_csv(touch_file, sep=',', skiprows=1, header=None)
            (fs, frame) = sciw.read(audio_file, mmap=False)
            feature_matrix = create_feature_vector(touch_features, frame, fs, mode)

            with open(TRAIN_FEATURES_FILE, 'a') as f_handle:
                np.savetxt(f_handle, feature_matrix, delimiter=',')

            with open(TRAIN_LABELS_FILE, 'a') as f_handle:
                np.savetxt(f_handle, [1] if label == 'knuckle' else [-1], delimiter=',')
