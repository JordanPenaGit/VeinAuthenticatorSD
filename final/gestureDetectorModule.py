import cv2
import time

from vimba import *
import vimbaCamModule as vcm

DEBUGGING = True
MSG_ON_SCANNING = "Align Blocks"


class gesturesDetector:
    def __init__(self, sleepAmount=1, captureTime=2, cameraWidth=640, cameraHeight=480):
        """
        This function is called when the class is instantiated. It sets the sleep amount and capture time
        for the class

        :param sleepAmount: This is the amount of time in seconds that the program will sleep for, defaults
        to 1 (optional)
        :param captureTime: The amount of time to capture an image, defaults to 2 (optional)
        """
        self.SLEEP_AMOUNT = sleepAmount
        self.SECONDS_TO_CAPTURE_IMAGE = captureTime
        self.timePassed = 0
        self.QR_CODE_ACTIVATED = False
        self.CAMERA_PIXEL_WIDTH = cameraWidth
        self.CAMERA_PIXEL_HEIGHT = cameraHeight

    def system_to_sleep(self):
        """
        The function system_to_sleep() is called when the user's thumb is not detected by the camera.
        The function prints a message to the user, resets the timer, and puts the system to sleep for a
        specified amount of time
        """
        self.timePassed = 0
        print(
            "\nThumb is deactivated.\nTimer reset.\nEntering sleep mode for "
            + str(self.SLEEP_AMOUNT)
            + " second..."
        )
        print("-" * 30)
        time.sleep(self.SLEEP_AMOUNT)

    def displayLocation(self, img, lm1, displayBoundaries):
        if displayBoundaries:
            # draw rectangle around index finger
            lm1x, lm1y = lm1[1:]

            img[:] = (0, 0, 0)

            cv2.rectangle(
                img=img,
                pt1=(int(lm1x - 50), int(lm1y - 50)),
                pt2=(int(lm1x + 50), int(lm1y + 25)),
                color=(255, 255, 0),
                thickness=7,
                lineType=cv2.LINE_AA,
            )

            cv2.rectangle(
                img=img,
                pt1=(
                    int(self.CAMERA_PIXEL_WIDTH * 0.41),
                    int(self.CAMERA_PIXEL_HEIGHT * 0.34),
                ),
                pt2=(
                    int(self.CAMERA_PIXEL_WIDTH * 0.48),
                    int(self.CAMERA_PIXEL_HEIGHT * 0.43),
                ),
                color=(255, 0, 255),
                thickness=7,
                lineType=cv2.LINE_AA,
            )

            cv2.putText(
                img=img,
                text=MSG_ON_SCANNING,
                org=(200, 100),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=4,
                color=(255, 255, 255),
                thickness=3,
                lineType=cv2.LINE_AA,
            )

    def captureImageTimerStart(self):
        if self.timePassed == 0:
            self.timePassed = time.time()
            print(
                "\nEncapsulating Index finger.\nPlease wait "
                + str(self.SECONDS_TO_CAPTURE_IMAGE)
                + " seconds..."
            )

        # Time is measure in seconds
        if time.time() - self.timePassed >= self.SECONDS_TO_CAPTURE_IMAGE:
            self.resetCaptureTimer()
            print(
                "\nTimer exceeded the "
                + str(self.SECONDS_TO_CAPTURE_IMAGE)
                + " seconds Index finger image capture start..."
            )

            return True

        return False

    def captureImage(self, img, fileStoreLocation):
        """
        The function starts the process of capturing an image from the Sensor ("Vimba"), and then returns the
        original image and the Sensor image

        :param img: The image of the index finger
        :return: a tuple of the original image and the sensor image.
        """
        sensorImg = img

        # Capture and crop img of the index for later processing.
        with Vimba.get_instance() as vimba:
            cams = vimba.get_all_cameras()
            with cams[0] as sensor:
                handler = vcm.Handler(sensor=sensor, filePath=fileStoreLocation)
                handler.setup_sensor()

                try:
                    # Add for loop to capture multiple images.
                    sensorImg = sensor.start_streaming(handler=handler, buffer_count=10)
                    handler.shutdown_event.wait()
                finally:
                    sensor.stop_streaming()

        time.sleep(self.SLEEP_AMOUNT)
        print("Sensor image capture..." + "\n" + "-" * 30)

        return (img, sensorImg)

    def resetCaptureTimer(self):
        """
        This function resets the time passed since the last capture to 0
        """
        self.timePassed = 0

    def activateQRCode(self):
        self.QR_CODE_ACTIVATED = True

    def deActivateQRCode(self):
        self.QR_CODE_ACTIVATED = False

    def getQRStatus(self):
        if self.QR_CODE_ACTIVATED:
            return True

        return False
