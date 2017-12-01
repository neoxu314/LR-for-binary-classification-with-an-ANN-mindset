
import numpy as np
import cv2
import h5py
import glob
from sklearn.model_selection import train_test_split


map_characters = {0: 'tree_1', 1: 'tree_2'}

pic_size = 64
batch_size = 32
epochs = 200
num_classes = len(map_characters)
pictures_per_class = 1000
test_size = 0.15


# load_pictures(): load pictures and labels from the characters folder
def load_pictures(BGR):
    """
    Load pictures from folders for characters from the map_characters dict and create a numpy dataset and
    a numpy labels set. Pictures are re-sized into picture_size square.
    :param BGR: boolean to use true color for the picture (RGB instead of BGR for plt)
    :return: dataset, labels set
    """
    pics = []
    labels = []

    # for loops: https://wiki.python.org/moin/ForLoop
    # dict.items(): \
    # https://stackoverflow.com/questions/10458437/what-is-the-difference-between-dict-items-and-dict-iteritems
    # this for loop is used for: the traversal of the map_characters, k is the key and char is the value
    for k, char in map_characters.items():
        # print(k, char)

        # glob module: https://docs.python.org/3.5/library/glob.html
        # k for k in x (List Comprehensions): https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions
        # picture is a array of the picture names in target folder
        # pictures = [k for k in glob.glob('./characters/%s/*' % char)]
        pictures = [k for k in glob.glob('./frames/%s/*' % char)]
        # print(pictures)

        # nb_pic: the number of the pictures array
        # https://stackoverflow.com/questions/2529536/python-idiom-for-if-else-expression
        nb_pic = round(pictures_per_class/(1-test_size)) if round(pictures_per_class/(1-test_size))<len(pictures) else len(pictures)
        nb_pic_int = int(nb_pic)
        # nb_pic = len(pictures)
        # print(nb_pic)

        # np.random.choice(pictures, nb_pic): use nb_pic to randomly generate a \
        # array consists of the nb_picture elements of pictures
        # print(np.random.choice(pictures, nb_pic))
        # np.random.choice(): https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.choice.html
        # pic is the relative path of an image
        for pic in np.random.choice(pictures, nb_pic_int):
            # cv2.imread(): read the image
            # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_image_display/py_image_display.html
            # a.shape is (width, height, colour?)
            a = cv2.imread(pic)
            if BGR:
                a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)

            # resize every image to pic_size*pic_size
            a = cv2.resize(a, (pic_size, pic_size))

            # add a to pics array
            # add k to label array, k is from 0 to 17
            pics.append(a)
            labels.append(k)

    return np.array(pics), np.array(labels)


def get_dataset(save=False, load=False, BGR=False):
    """
    get X_train, X_test, y_train, y_test

    Create the actual dataset split into train and test, pictures content is as float32 and
    normalized (/255.). The dataset could be saved or loaded from h5 files.
    :param save: saving or not the created dataset
    :param load: loading or not the dataset
    :param BGR: boolean to use true color for the picture (RGB instead of BGR for plt)
    :return: X_train, X_test, y_train, y_test (numpy arrays)
    """
    if load:
        # load data from h5py file

        # open and read the training and testing image data from dataset.h5
        h5f = h5py.File('dataset.h5', 'r')
        # Python slice notation: https://stackoverflow.com/questions/509211/explain-slice-notation
        # a[:]: a copy of the whole array
        X_train = h5f['X_train'][:]
        X_test = h5f['X_test'][:]
        #close the file
        h5f.close()

        # open and read the training and testing label data from labels.h5
        h5f = h5py.File('labels.h5', 'r')
        y_train = h5f['y_train'][:]
        y_test = h5f['y_test'][:]
        h5f.close()
    else:
        # load data from image folder

        # X is picture array of the training set, y is the labels array of the training set
        X, y = load_pictures(BGR)

        # to_categorical: https://keras.io/utils/#to_categorical
        # Converts a class vector (integers) y to binary class matrix.
        # y = keras.utils.to_categorical(y, num_classes)

        # train_test_split: Split arrays or matrices into random train and test subsets
        # X_train and x_test are split from X, y_train and y_test are split from y
        # if the type of the test_size is float \
        # it represents the proportion of the dataset to include in the test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        if save:
            # save X_train, X_test to the dataset.h5
            h5f = h5py.File('dataset.h5', 'w')
            h5f.create_dataset('X_train', data=X_train)
            h5f.create_dataset('X_test', data=X_test)
            h5f.close()

            # y_train, y_test to the labels.h5
            h5f = h5py.File('labels.h5', 'w')
            h5f.create_dataset('y_train', data=y_train)
            h5f.create_dataset('y_test', data=y_test)
            h5f.close()

    # Feature normalisation of X_train and X_test
    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    print("Train", X_train.shape, y_train.shape)
    print("Test", X_test.shape, y_test.shape)

    return X_train, X_test, y_train, y_test
