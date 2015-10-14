import csv
import os
from fileutils import PROJECT_DIR, TEST_DATA_DIR
import pandas as pd
import scipy.io.wavfile as sciw
from feature_extraction import create_feature_vector
import numpy as np

FINAL_PREDICTIONS = os.path.join(PROJECT_DIR, 'fingersense-test-labels.csv')

def predict_test_labels(classifier, std_scale, pca_std):
    """
    This function predicts labels for test data using the trained classifier and writes output to file
    :param classifier: classifier trained on train data
    :param std_scale
    :param pca_std

    """

    top_level_test_dirs = os.listdir(TEST_DATA_DIR)
    top_level_test_dirs = [x for x in top_level_test_dirs if not x.startswith('.')]

    predictions_list = [['timestamp', 'label']]

    # Remove prediction data file if exists
    try:
        os.remove(FINAL_PREDICTIONS)
    except OSError:
        pass

    # Walks through directories in test directory and reads touch.csv, audio.wav files
    # Extracts timestamp and mode from directory names
    for dir_name in top_level_test_dirs:
        mode = dir_name.split('-')[0]
        timestamped_dirs = os.listdir(os.path.join(TEST_DATA_DIR, dir_name))
        timestamped_dirs = [x for x in timestamped_dirs if not x.startswith('.')]
        for folder_name in timestamped_dirs:
            timestamp = folder_name
            audio_file = os.path.join(TEST_DATA_DIR, dir_name, folder_name, 'audio.wav')
            touch_file = os.path.join(TEST_DATA_DIR, dir_name, folder_name, 'touch.csv')
            touch_features = pd.read_csv(touch_file, sep=',', skiprows=1, header=None)
            (fs, frame) = sciw.read(audio_file, mmap=False)

            # Creating test feature matrix
            feature_matrix = create_feature_vector(touch_features, frame, fs, mode)

            # Transform each test feature into the same space as training data
            test_feature = pca_std.transform(std_scale.transform(feature_matrix))

            # Predict labels for each test feature and write string mappings of labels
            label = 'knuckle' if classifier.predict(test_feature) == np.array([1]) else 'pad'
            predictions_list.append([timestamp, label])

    # Write predictions to the output file
    with open(FINAL_PREDICTIONS, 'w') as pred_fh:
        writer = csv.writer(pred_fh, delimiter=',')
        writer.writerows(predictions_list)