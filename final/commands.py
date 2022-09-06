import os
import cv2
import time
import threading
import numpy as np
import handTrackingModule as htm
import gestureDetectorModule as gdm

from keras import backend as K
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D, Input
from keras.layers.core import Lambda, Flatten, Dense
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2

from PIL import Image, ImageEnhance
from pyzbar.pyzbar import decode

HALF_CAMERA_RESOLUTION = False
DISPLAY_MENU_OPTIONS = True
CAMERA_USB_LOCATION = 0
CAMERA_PIXEL_WIDTH = 1920
CAMERA_PIXEL_HEIGHT = 1080

if HALF_CAMERA_RESOLUTION:
    CAMERA_PIXEL_WIDTH = 960
    CAMERA_PIXEL_HEIGHT = 540


HAND_CONFIDENCE_VALUE = 0.65
QR_PASSWORD = b"SD_Final_Presentation"

DISPLAY_BOUNDARIES = True
NUMBER_OF_IMGS_CAPTURE_WITH_SENSOR = 5

MENU_TIMER = 0
SWAP_MENU_IMG_TIMER = 2
CAPTURE_IMGS_TIMER = 5

FILE_NEW_IMG = "./final/images/newImgs/"
USERS_IMG_DIRECTORY = "./final/images/usersId/"
MENU_IMG_FOLDER_PATH = "./final/menuImages/"
WEIGHTS_FILE_PATH = "./final/weights.20000.h5"

THREAD_TASK = 0
USER_FAILED = False
USER_SUCCESS = False
THREAD_TASK_ACTIVE = False
MODEL_PROCESSING = False

LM1 = 10
LM2 = 15


def menuOptionsDisplay():
    """
    If the user has not yet made a selection, display the first menu image. If the user has made a
    selection, display the second menu image. If the user has made a selection and it was successful,
    display the third menu image. If the user has made a selection and it was unsuccessful, display the
    fourth menu image

    :return: the index of the image to be display
    """
    global MENU_TIMER
    global USER_SUCCESS
    global USER_FAILED

    if MENU_TIMER == 0:
        MENU_TIMER = time.time()
        return 3

    if time.time() - MENU_TIMER >= (SWAP_MENU_IMG_TIMER * 2):
        MENU_TIMER = 0
        return 3

    if time.time() - MENU_TIMER >= SWAP_MENU_IMG_TIMER:
        USER_SUCCESS = False
        USER_FAILED = False
        return 0

    if USER_SUCCESS:
        return 1

    if USER_FAILED:
        return 2

    return 3


def displayMenuImages(img):
    if HALF_CAMERA_RESOLUTION:
        print(img.shape)
        img[0:540, 0:960] = menuImgs[menuOptionsDisplay()]
    else:
        img[0:1080, 0:1920] = menuImgs[menuOptionsDisplay()]


def create_new_user(img, gestures):
    """
    The function takes in an image and a gesture object and captures a number of images (defined by
    the global variable NUMBER_OF_IMGS_CAPTURE_WITH_SENSOR) and stores them in the directory defined by
    the global variable USERS_IMG_DIRECTORY

    :param img: The image to be captured
    :param gestures: The Gestures object that we created earlier
    """
    global USERS_IMG_DIRECTORY
    global NUMBER_OF_IMGS_CAPTURE_WITH_SENSOR
    global THREAD_TASK_ACTIVE

    print("Capturing new users images...")

    for x in range(NUMBER_OF_IMGS_CAPTURE_WITH_SENSOR):
        gestures.captureImage(img=img, fileStoreLocation=USERS_IMG_DIRECTORY)

    print("New users store...")
    THREAD_TASK_ACTIVE = False


def init_user_validation(img, model, gestures):
    """
    The function takes in an image, a model, and a gesture object. It then captures a number of images
    from the gesture object, and compares them to the images in the user's directory. If the images are
    similar enough, the user is validated

    :param img: The image that will be captured by the camera
    :param model: The model that we created in the previous step
    :param gestures: This is the object that we created in the previous step
    """
    global THREAD_TASK_ACTIVE
    global USER_SUCCESS
    global USER_FAILED
    global FILE_NEW_IMG
    global NUMBER_OF_IMGS_CAPTURE_WITH_SENSOR

    for x in range(NUMBER_OF_IMGS_CAPTURE_WITH_SENSOR):
        gestures.captureImage(img=img, fileStoreLocation=FILE_NEW_IMG)

    finalArr = []

    for file in os.listdir(path=FILE_NEW_IMG):
        fileName = os.fsdecode(filename=file)
        resultArr = []

        compareImg = cv2.imread(
            filename=(f"{FILE_NEW_IMG}{fileName}"),
            flags=cv2.IMREAD_COLOR,
        )
        print(
            "\n\n"
            + "+" * 50
            + "\nIn usersId comparing file: "
            + str(fileName)
            + "\n"
            + "+" * 50
        )
        counter = 0
        counter2 = 0
        folder = os.listdir(path=USERS_IMG_DIRECTORY)
        for newImg in folder:
            newImgName = os.fsdecode(filename=newImg)
            if newImgName == ".DS_Store":
                continue
            else:
                tempImg = cv2.imread(filename=f"{USERS_IMG_DIRECTORY}{newImgName}")

                mlResult = predict(model=model, i1=tempImg, i2=compareImg)
                resultArr.append(mlResult[0][0])
                print("counter is: ", counter, "counter2 is:", counter2)

                if counter % 4 == 0 and counter != 0:
                    counter2 += 1
                    print("Checking now...")
                    arrMean = np.mean(resultArr)
                    finalArr.append(arrMean)
                    print(np.max(finalArr), " ", arrMean)
                    if np.max(finalArr) >= 0.75:
                        USER_SUCCESS = True
                        break
                    else:
                        if counter2 == len(folder):
                            break
                        finalArr = []
                        counter = 0
                        continue
                counter += 1
                counter2 += 1

        # arrMean = np.mean(resultArr)
        # finalArr.append(arrMean)
        print("\n\n" + "+" * 50 + "\narrMean values: " + str(arrMean) + "\n" + "+" * 50)
    print("out of loop")
    for f in os.listdir(path=FILE_NEW_IMG):
        os.remove(os.path.join(FILE_NEW_IMG, f))

    print(
        "\n\n"
        + "+" * 50
        + "\nfinalArr values: "
        + str(np.max(finalArr))
        + "\n"
        + "+" * 50
    )

    if np.max(finalArr) >= 0.75:
        USER_SUCCESS = True
        print("You shall pass.")

    else:
        USER_FAILED = True
        print("You shall not pass.")

    THREAD_TASK_ACTIVE = False


def initialize_weights(shape, dtype=None):
    """
    It returns a tensor of the specified shape sampled from a zero mean normal distribution with a
    standard deviation of 0.01

    :param shape: The shape of the tensor to initialize
    :param dtype: The data type expected by the input, as a string (float32, float64, etc.)
    :return: A random normal distribution with a mean of 0 and a standard deviation of 0.01
    """
    return np.random.normal(loc=0.0, scale=1e-2, size=shape)


def initialize_bias(shape, dtype=None):
    """
    It returns a tensor of the specified shape sampled from a normal distribution with mean 0.5 and
    standard deviation 0.01

    :param shape: The shape of the bias tensor
    :param dtype: The data type expected by the input, as a string (float32, float64, etc.)
    :return: A random number between 0.5 and 0.5 + 1e-2
    """
    return np.random.normal(loc=0.5, scale=1e-2, size=shape)


def get_siamese_model(input_shape):
    """
    It takes two images as input, passes them through a CNN, and returns the L1 distance between the two
    encodings

    :param input_shape: The shape of the input images
    :return: A model that takes two inputs and returns a single output.
    """

    # Model architecture based on the one provided in: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf

    left_input = Input(input_shape)
    right_input = Input(input_shape)

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

    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    return siamese_net


def image_process3(im):
    """
    We take the image, enhance the contrast, resize it, and then apply a Canny edge detection
    algorithm to it

    :param im: the image to be processed
    :return: A numpy array of the image
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


def predict(model, i1, i2):
    """
    It takes two images, processes them, and then feeds them into the model

    :param model: the model we just created
    :param i1: path to the first image
    :param i2: the image you want to compare to
    :return: The similarity score between the two images.
    """

    img1 = image_process3(i1)
    img2 = image_process3(i2)

    validation_x = np.asarray(img1)
    validation_y = np.asarray(img2)

    validation_x = validation_x.reshape(1, 128, 128, 1)
    validation_y = validation_y.reshape(1, 128, 128, 1)

    input = validation_x, validation_y
    score = model.predict(input)
    print("similarity score: ", score)

    return score


# Load imgs from the menuImages folder
menuImgs = []
menuImgsLocation = os.listdir(MENU_IMG_FOLDER_PATH)
for imPath in menuImgsLocation:
    menuImg = cv2.imread(filename=f"{MENU_IMG_FOLDER_PATH}{imPath}")

    if HALF_CAMERA_RESOLUTION:
        menuImg = cv2.resize(src=menuImg, dsize=(0, 0), fx=0.5, fy=0.5)

    menuImgs.append(menuImg)

cap = cv2.VideoCapture(CAMERA_USB_LOCATION)
cap.set(3, CAMERA_PIXEL_WIDTH)
cap.set(4, CAMERA_PIXEL_HEIGHT)

detector = htm.handDetector(detectionCon=HAND_CONFIDENCE_VALUE)
gestures = gdm.gesturesDetector(
    sleepAmount=1,
    captureTime=CAPTURE_IMGS_TIMER,
    cameraWidth=CAMERA_PIXEL_WIDTH,
    cameraHeight=CAMERA_PIXEL_HEIGHT,
)

model = get_siamese_model((128, 128, 1))
model.load_weights(WEIGHTS_FILE_PATH)


# MAIN while loop
while True:
    success, img = cap.read()
    # img = cv2.flip(src=img, flipCode=1)
    img = detector.findHands(img=img, draw=False)

    decodeObj = decode(image=img)

    if not (len(decodeObj) == 0):
        if decodeObj[0].data == QR_PASSWORD:
            print("QR Code Scan...")
            gestures.activateQRCode()

    lmList = detector.findPosition(img=img, draw=False)

    if len(lmList) == 0:  # If no hand is visible by the webcam
        if THREAD_TASK_ACTIVE:
            # Add message waiting for result.
            pass

        else:
            if DISPLAY_MENU_OPTIONS:
                displayMenuImages(img)

            gestures.resetCaptureTimer()

    if len(lmList) != 0:  # If a hand is visible by the webcam
        gestures.displayLocation(
            img=img,
            lm1=lmList[LM1],
            displayBoundaries=DISPLAY_BOUNDARIES,
        )
        fingers = detector.fingersUp()

        if THREAD_TASK_ACTIVE:
            pass

        elif (
            fingers[0] == 1
            and fingers[1] == 1
            and fingers[2] == 1
            and fingers[3] == 1
            and fingers[4] == 1
        ):  # all fingers exposed.
            print("+++++++++++++Sign Up")
            if gestures.captureImageTimerStart() and gestures.getQRStatus():
                print("New user sign up starting...")
                gestures.deActivateQRCode()
                THREAD_TASK_ACTIVE = True
                THREAD_TASK = threading.Thread(
                    target=create_new_user, args=(img, gestures)
                )
                THREAD_TASK.start()
                # create_new_user(img, gestures)

        elif (
            fingers[0] == 0
            and fingers[1] == 1
            and fingers[2] == 1
            and fingers[3] == 0
            and fingers[4] == 0
        ):
            # Index and middle
            print("Log In")
            if gestures.captureImageTimerStart():
                gestures.deActivateQRCode()
                THREAD_TASK_ACTIVE = True
                THREAD_TASK = threading.Thread(
                    target=init_user_validation, args=(img, model, gestures)
                )
                THREAD_TASK.start()

    else:
        gestures.resetCaptureTimer()

    cv2.imshow(winname="WebCam", mat=img)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
