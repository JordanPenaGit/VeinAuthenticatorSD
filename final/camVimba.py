import threading
import cv2
from vimba import *
import vimbaCamModule as vcm


# It creates a window for each camera and displays the frames as they arrive
class Handler:
    """
    Class that handles the frame processing and passing to cv2.imshow().

    Also, manages the shutdown of the window / program.

    :TEMPORARY: When shutdown is run it process the image color conversion.
    """

    def __init__(self):
        self.shutdown_event = threading.Event()

    def __call__(self, cam: Camera, frame: Frame):
        ENTER_KEY_CODE = 13

        key = cv2.waitKey(1)  # returns the ing value of key press.
        if key == ENTER_KEY_CODE:
            self.shutdown_event.set()
            cvtFrame(img=frame.as_opencv_image())
            return

        elif frame.get_status() == FrameStatus.Complete:
            cv2.imshow(
                winname="Stream. Press <Enter> to stop stream.",
                mat=frame.as_opencv_image(),
            )

        cam.queue_frame(frame=frame)


def cvtFrame(img):
    image = cv2.cvtColor(src=img, code=cv2.COLOR_BAYER_GR2RGB)
    cv2.imwrite(filename="frame3.jpg", img=image)


def setup_camera(cam: Camera):
    """
    The function sets the camera to a monochrome format that is compatible with OpenCV

    :param cam: Camera: The camera object that we are setting up
    :type cam: Camera
    """
    with cam:
        try:
            cam.ExposureAuto.set("Once")

        except (AttributeError, VimbaFeatureError):
            pass

        try:
            cam.BalanceWhiteAuto.set("Once")

        except (AttributeError, VimbaFeatureError):
            pass

        # Query available, open_cv compatible pixel formats
        # prefer color formats over monochrome formats
        cv_fmts = intersect_pixel_formats(
            fmts1=cam.get_pixel_formats(), fmts2=OPENCV_PIXEL_FORMATS
        )  # returns all the possible color formats available for OpenCV. Formats that are color or mono

        mono_fmts = intersect_pixel_formats(fmts1=cv_fmts, fmts2=MONO_PIXEL_FORMATS)
        cam.set_pixel_format(fmt=mono_fmts[0])


def main():
    with Vimba.get_instance() as vimba:
        cams = vimba.get_all_cameras()
        with cams[0] as cam:

            # Start Streaming, wait for five seconds, stop streaming
            # handler = Handler()
            # setup_camera(cam)

            handler2 = vcm.Handler(sensor=cam)
            handler2.setup_sensor()

            try:
                # Start Streaming with a custom a buffer of 10 Frames (defaults to 5)
                cam.start_streaming(handler=handler2, buffer_count=30)
                handler2.shutdown_event.wait()

            finally:
                cam.stop_streaming()


if __name__ == "__main__":
    main()
