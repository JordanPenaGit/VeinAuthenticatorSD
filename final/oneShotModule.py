import cv2
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

from keras import backend as K
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D, Input
from keras.layers.core import Lambda, Flatten, Dense
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2

WEIGHTS_FILE_PATH = "./final/weights.20000.h5"


class OneShot:
    def __init__(self):
        """
        It takes in an image, resizes it to 128x128, converts it to grayscale, and then returns the image as
        a numpy array
        """
        tensor = (128, 128, 1)
        self.activeModel = self.get_siamese_model(tensor)
        self.activeModel.load_weights(filepath=WEIGHTS_FILE_PATH)

    def get_siamese_model(self, input_shape):
        """
        The function takes in two images, and returns a single value that indicates whether or not the
        two images are of the same person

        :param input_shape: The shape of the input images
        :return: The model is being returned.
        """

        # Model architecture based on the one provided in: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        # Define the tensors for the two input images
        left_input = Input(input_shape)
        right_input = Input(input_shape)

        # Convolutional Neural Network
        model = Sequential()
        model.add(
            Conv2D(
                64,
                (10, 10),
                activation="relu",
                input_shape=input_shape,
                kernel_initializer=initialize_weights,
                kernel_regularizer=l2(2e-4),
            )
        )
        model.add(MaxPooling2D())
        model.add(
            Conv2D(
                128,
                (7, 7),
                activation="relu",
                kernel_initializer=initialize_weights,
                bias_initializer=initialize_bias,
                kernel_regularizer=l2(2e-4),
            )
        )
        model.add(MaxPooling2D())
        model.add(
            Conv2D(
                128,
                (4, 4),
                activation="relu",
                kernel_initializer=initialize_weights,
                bias_initializer=initialize_bias,
                kernel_regularizer=l2(2e-4),
            )
        )
        model.add(MaxPooling2D())
        model.add(
            Conv2D(
                256,
                (4, 4),
                activation="relu",
                kernel_initializer=initialize_weights,
                bias_initializer=initialize_bias,
                kernel_regularizer=l2(2e-4),
            )
        )
        model.add(Flatten())
        model.add(
            Dense(
                4096,
                activation="sigmoid",
                kernel_regularizer=l2(1e-3),
                kernel_initializer=initialize_weights,
                bias_initializer=initialize_bias,
            )
        )

        encoded_l = model(left_input)
        encoded_r = model(right_input)

        L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([encoded_l, encoded_r])

        prediction = Dense(1, activation="sigmoid", bias_initializer=initialize_bias)(
            L1_distance
        )

        return Model(inputs=[left_input, right_input], outputs=prediction)

    def image_process_right(self, im):
        """
        It takes an image, enhances the contrast, resizes it, applies a canny edge detection, dialates the
        image, and then returns the image

        :param im: The image to be processed
        :return: The image is being returned after being processed.
        """

        im = Image.fromarray(obj=np.uint8(im))
        new_im = ImageEnhance.Contrast(image=im)
        new_im.enhance(factor=1.3)
        new_im = np.resize(a=im, new_shape=(128, 128))

        npArray = asarray(new_im)
        kernel = np.ones((5, 5), np.uint8)

        edges = cv2.Canny(npArray, 20, 100)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.dilate(npArray, kernel, iterations=1)
        # edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(
            edges,
            255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=15,
            C=0,
        )

        return thresh

    def image_process_left(self, im):
        """
        It takes an image, enhances the contrast, resizes it, applies a canny edge detection, dialates the
        image, and then returns the image

        :param im: The image to be processed
        :return: The image is being returned after being processed.
        """

        im = Image.fromarray(np.uint8(im))
        new_im = ImageEnhance.Contrast(im)
        new_im.enhance(1.3)
        new_im = im.resize((128, 128))

        numpyArray = np.asarray(new_im)
        kernel = np.ones((5, 5), np.uint8)

        edges = cv2.Canny(numpyArray, 20, 100)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.dilate(numpyArray, kernel, iterations=1)
        edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(
            edges,
            255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=15,
            C=0,
        )

        return thresh

    def predict(self, i1, i2):
        """
        The function takes in two images, processes them, and then feeds them into the model to get a
        similarity score.

        :param i1: the first image
        :param i2: the image to be compared to the reference image
        :return: The similarity score between the two images.
        """

        img1 = self.image_process_left(asarray(i1))
        img2 = self.image_process_right(asarray(i2))

        validation_x = np.asarray(img1)
        validation_y = np.asarray(img2)

        validation_x = validation_x.reshape(1, 128, 128, 1)
        validation_y = validation_y.reshape(1, 128, 128, 1)

        # self.activeModel.summary()

        input = (validation_x, validation_y)
        score = self.activeModel.predict(input)
        print("similarity score: ", score)

        return score


def initialize_weights(shape, dtype=None):
    """
    It returns a tensor of the specified shape initialized with random normal values

    :param shape: The shape of the tensor to initialize
    :param dtype: The data type expected by the input, as a string (float32, float64, etc.)
    :return: A random normal distribution with a mean of 0 and a standard deviation of 0.01
    """
    return np.random.normal(loc=0.0, scale=1e-2, size=shape)


def initialize_bias(shape, dtype=None):
    """
    It returns a tensor of the given shape filled with random normal values

    :param shape: The shape of the tensor to initialize
    :param dtype: The data type expected by the input, as a string (float32, float64, etc.)
    :return: A random number between 0.5 and 0.5 + 1e-2
    """
    return np.random.normal(loc=0.5, scale=1e-2, size=shape)


def main(i1, i2):
    model = OneShot()
    return model.predict(i1=i1, i2=i2)


if __name__ == "__main__":
    main(
        cv2.imread(
            filename="./final/images/signIn/24-06-2022-19-23-21.png",
            flags=cv2.IMREAD_GRAYSCALE,
        ),
        cv2.imread(
            filename="./final/images/signIn/24-06-2022-19-23-21.png",
            flags=cv2.IMREAD_GRAYSCALE,
        ),
    )  # add parse values
