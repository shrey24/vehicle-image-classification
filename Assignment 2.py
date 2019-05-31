import os
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.feature import hog as sk_hog
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn import metrics


""" Utility functions to display images"""


def display_one(a, title1="Original"):
    # Display one image
    plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.show()


def display(a, b, title1="Original", title2="Edited"):
    # Display two images
    plt.figure()
    plt.subplot(121), plt.imshow(a), plt.title(title1)
#     plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(b), plt.title(title2)
#     plt.xticks([]), plt.yticks([])
    plt.grid(True)
    plt.show()  


def read_labels(file_path):
    y_train = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            y_train.append(line)
    return y_train


"""Extract images from folders, and rescale it to 64X64"""


def loadImagePaths(path):
    '''function to load folder into arrays and
    then it returns that same array'''
    # Put files into lists and return them as one list
    image_files = sorted( [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.jpg')] )
    return image_files


def scale_imgs(cv2_img_list, height, width):
    dim = (width, height)
    res_img = []
    for i in range(len(cv2_img_list)):
        res = cv2.resize(cv2_img_list[i], dim, interpolation=cv2.INTER_LINEAR)
        res_img.append(res)
    return res_img


def load_img_dataset(path_to_folder):
    X_img_paths = loadImagePaths(path_to_folder)
    X_images = [cv2.imread(img, cv2.IMREAD_UNCHANGED) for img in X_img_paths[:]]
    X_img_rescaled = scale_imgs(X_images, 64, 64)
    X_img_rescaled = np.array(X_img_rescaled)
    return X_img_rescaled


""" Extract Features """

# ### 1. HOG
def calc_hog(img): # hog_feature_vectors
    return sk_hog(img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), multichannel=True)


# ## DR

def svd_transform(feature_dataset, components):
    svd = TruncatedSVD(n_components=components)
    svd.fit(feature_dataset)
    return svd.transform(feature_dataset)


if __name__ == '__main__':

    X_train_imgs_path = 'traffic_data/traffic/train/'
    y_train_labels_path = 'traffic_data/traffic/train.labels'
    X_test_imgs_path = 'traffic_data/traffic/test/'

    # change this as per requirement
    _train_feature_txt = 'train_features.txt'
    _test_feature_txt = 'test_features.txt'
    _output_file = 'test.txt'

    # Extract labels from train.labels and test.labels
    y_train = np.loadtxt(y_train_labels_path)
    t0_read = time.time()
    if os.path.isfile(_train_feature_txt) and os.path.isfile(_train_feature_txt):
        print("reading features from ", _train_feature_txt, _test_feature_txt)
        X_train_hog = np.loadtxt(_train_feature_txt, delimiter=",")
        X_test_hog = np.loadtxt(_test_feature_txt, delimiter=",")


    else:
        print("loading data...")
        # Load the dataset of images
        X_train = load_img_dataset(X_train_imgs_path)
        X_test = load_img_dataset(X_test_imgs_path)
        print("loaded train images: ", X_train.shape)
        print("loaded test images: ", X_test.shape)

        X_train_hog = np.array([calc_hog(img) for img in X_train])
        X_test_hog = np.array([calc_hog(img) for img in X_test])
        print("HoG features extracted: ")
        print("X train(hog) shape: ", X_train_hog.shape)
        print("saving features to ", _train_feature_txt, _test_feature_txt)
        np.savetxt(_train_feature_txt, X_train_hog, delimiter=",")
        np.savetxt(_test_feature_txt, X_test_hog, delimiter=",")

    t1_read = time.time()
    print("time taken in reading data: ", t1_read-t0_read)
    print("Applying model..", '*'*20)

    # print("applying kNN", '='*20)
    # # using hog features only (512,) dim per img
    # knn_classifier = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
    # knn_classifier.fit(X_train_hog, y_train)
    #
    # y_pred = knn_classifier.predict(X_test_hog)

    # print("applying SVM", '=' * 20)
    # svm_classifier = svm.SVC(gamma='scale', decision_function_shape='ovo')
    # svm_classifier.fit(X_train_hog, y_train)
    #
    # y_pred = svm_classifier.predict(X_test_hog)
    time0 = time.time()
    rand_forest = RandomForestClassifier(n_estimators=50, max_depth=2, random_state=42, criterion='gini')
    rand_forest.fit(X_train_hog, y_train)
    time1 = time.time()
    print("time taken in learning: ", time1 - time0)

    y_pred = rand_forest.predict(X_test_hog)
    time2 = time.time()
    print("time taken in prediction: ", time2 - time1)
    # save results
    np.savetxt(_output_file, y_pred, fmt='%d', delimiter="\n")
    print("Prediction completed. results saved to ", _output_file)

