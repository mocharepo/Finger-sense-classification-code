from assemble_data import prepare_training_data
from classify import build_classifier
from predict_labels import predict_test_labels

if __name__ == '__main__':
    # Walks through training data directories and extracts features of interest
    print 'Data preparation in progress ...'
    prepare_training_data()

    # Trains and builds SVM classifier
    print 'Building classifier ...'
    classifier, std_scale, pca_std = build_classifier()

    # Reads test data, predicts labels
    print 'Predicting labels for test data ...'
    predict_test_labels(classifier, std_scale, pca_std)

    print 'Done.'
