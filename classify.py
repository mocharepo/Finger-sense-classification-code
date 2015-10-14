from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
import pandas as pd
from fileutils import PROJECT_DIR
import os

def build_classifier():
    """
    This function builds classifier using training data and labels
    :return:
        clf:        classifier
        std_scale:  Normalization parameters used for transforming training data
        pca_std:    Principal component analysis transformation parameters
    """
    X = pd.read_csv(os.path.join(PROJECT_DIR, "train_features.csv"), skiprows=1, header=None).as_matrix()
    Y = pd.read_csv(os.path.join(PROJECT_DIR, "train_labels.csv"), header=None).as_matrix().ravel()

    # Split data into training and cross validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=3131)

    std_scale = preprocessing.StandardScaler().fit(X_train)
    X_train_std = std_scale.transform(X_train)
    # X_test_std = std_scale.transform(X_test)

    pca_std = PCA(n_components=13).fit(X_train_std)
    X_train_std = pca_std.transform(X_train_std)
    # X_test_std = pca_std.transform(X_test_std)

    clf = svm.SVC(C=5)
    clf.fit(X_train_std, y_train)

    # Compare predictions of classifier on cross-validation sets with ground-truths
    # print clf.score(X_test_std, y_test)
    return clf, std_scale, pca_std
