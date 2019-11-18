# Import the modules
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np
from collections import Counter
from scipy.io import loadmat
from lightgbm import LGBMClassifier

# Load the dataset
# dataset = datasets.fetch_mldata("MNIST Original")
# https://www.kaggle.com/avnishnish/mnist-original
mnist = loadmat("../data/mnist-original.mat")
# dataset = datasets.load_digits()

# Extract the features and labels
# features = np.array(dataset.data, 'int16')
# labels = np.array(dataset.target, 'int')
features = mnist["data"].T
labels = mnist["label"][0]

# Extract the hog features
list_hog_fd = []
for feature in features:
    # fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

print("Count of digits in dataset", Counter(labels))

# Create an linear SVM object
# clf = LinearSVC()
clf = LGBMClassifier()

# Perform the training
clf.fit(hog_features, labels)

# Save the classifier
joblib.dump(clf, "../data/digits_cls_lgbm.pkl", compress=3)
