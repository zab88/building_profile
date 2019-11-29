# Import the modules
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np
from collections import Counter
from scipy.io import loadmat
from scipy import ndimage
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


# https://github.com/ChaitanyaBaweja/RotNIST/blob/master/images.py
'''
Augment training data with rotated digits
images: training images
labels: training labels
'''
def expand_training_data(images, labels):

    expanded_images = []
    expanded_labels = []
    # directory = os.path.dirname("data/New")
    # if not tf.gfile.Exists("data/New"):
    #     tf.gfile.MakeDirs("data/New")
    k = 0 # counter
    for x, y in zip(images, labels):
        #print(x.shape)
        k = k+1
        if k%100==0:
            print ('expanding data : %03d / %03d' % (k,np.size(images,0)))

        # register original data
        # expanded_images.append(x)
        # expanded_labels.append(y)

        bg_value = -0.5 # this is regarded as background's value black
        #print(x)
        image = np.reshape(x, (-1, 28))
        #time.sleep(3)
        #print(image)
        #time.sleep(3)
        # for i in range(9,):
        for i in [-15, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 15]:
        # for i in [-15, 15]:
            # rotate the image with random degree
            # angle = np.random.randint(-20,20,1)
            angle = i
            new_img_ = ndimage.rotate(image,angle,reshape=False, cval=bg_value)

            # shift the image with random distance
            shift = np.random.randint(-2, 2, 2)
            new_img_ = ndimage.shift(new_img_, shift, cval=bg_value)
            # new_img_ = ndimage.shift(new_img,shift, cval=bg_value)

            # code for saving some of these for visualization purpose only
            # image1 = (image*255) + (255 / 2.0)
            # new_img1 = (new_img_*255) + (255 / 2.0)
            new_img2 = np.reshape(new_img_,(28,28,1))
            #print(new_img1.shape)

            # register new training data

            expanded_images.append(new_img2.copy())
            expanded_labels.append(y)

    # return them as arrays

    expandedX=np.asarray(expanded_images)
    expandedY=np.asarray(expanded_labels)
    return expandedX, expandedY

# features, labels = expand_training_data(features[:6000], labels[:6000])
# features, labels = expand_training_data(features, labels)

# Extract the hog features
list_hog_fd = []
for feature in features:
    # fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    # feature = np.array([0 if x < 120 else 255 for x in feature])

    reshaped = feature.reshape((28, 28))
    # reshaped = np.array([0 if x < 50 else 255 for x in feature])
    # reshaped[reshaped < 50] = 0
    # reshaped[reshaped >= 50] = 255

    fd = hog(reshaped, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

print("Count of digits in dataset", Counter(labels))

# Create an linear SVM object
clf = LinearSVC()
# clf = LGBMClassifier(n_jobs=-1)

# Perform the training
clf.fit(hog_features, labels)

# Save the classifier
# joblib.dump(clf, "../data/digits_cls_bin_lgbm.pkl", compress=3)
joblib.dump(clf, "../data/digits_cls.pkl", compress=3)
