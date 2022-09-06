import cv2
import threading
import datetime
from vimba import *

IMG_WIDTH = 2952
IMG_HEIGHT = 1944


class Handler:
    def __init__(self, sensor, filePath):
        """
        Grabs and initializes the Vimba camera.

        :param camera: Vimba camera object
        """
        self.shutdown_event = threading.Event()
        self.cam = sensor
        self.filePath = filePath

    def __call__(self, cam: Camera, frame: Frame):
        """
        The function takes in a camera object and a frame object, and returns a sensor image

        :param cam: Camera
        :type cam: Camera
        :param frame: The frame object that is passed to the callback function
        :type frame: Frame

        :return: The sensor image is being returned.
        """
        self.shutdown_event.set()
        sensorImg = cvtFrame(
            img=frame.as_opencv_image(), file_store_location=self.filePath
        )
        print("Sensor image capture...")
        print("-" * 30)
        return sensorImg

    def setup_sensor(self):
        """
        The function sets the camera to a mono format, which is a format that is compatible with OpenCV
        """
        with self.cam:
            try:
                self.cam.ExposureAuto.set("Continuous")

            except (AttributeError, VimbaFeatureError):
                pass

            try:
                self.cam.BalanceWhiteAuto.set("Continuous")

            except (AttributeError, VimbaFeatureError):
                pass

            mono_fmts = intersect_pixel_formats(
                fmts1=self.cam.get_pixel_formats(), fmts2=MONO_PIXEL_FORMATS
            )  # Converts the Vimba frame to Mono8 format.

            self.cam.set_pixel_format(fmt=mono_fmts[0])


def cvtFrame(img, file_store_location):
    """
    It takes an Mono8 image, converts it to COLOR_BAYER_GR2RGB, saves it to a file, and returns the image.

    :param img: The Mono8 image.

    :return: Converted COLOR_BAYER_GR2RGB image.
    """
    global IMG_WIDTH
    global IMG_HEIGHT

    print("Capturing...")
    imgTitle = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    img = cv2.cvtColor(src=img, code=cv2.COLOR_BAYER_GR2RGB)

    crop_img = img[
        int(IMG_HEIGHT * 0.33) : int(IMG_HEIGHT * 0.65),
        int(IMG_WIDTH * 0.15) : int(IMG_WIDTH * 0.7),
    ]
    # print(crop_img.shape)

    cv2.imwrite(
        filename=f"{file_store_location} {imgTitle}.png",
        img=crop_img,
    )
    return img
