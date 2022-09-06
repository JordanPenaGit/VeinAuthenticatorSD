import mediapipe as mp
import cv2
import time


class handDetector:
    def __init__(
        self,
        mode=False,
        maxHands=2,
        modelComplexity=1,
        detectionCon=0.5,
        trackCon=0.5,
        chosenLm=0,
    ):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.chosenLm = chosenLm
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode,
            self.maxHands,
            self.modelComplexity,
            self.detectionCon,
            self.trackCon,
        )  # uses rgb imgs
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        """
        `findHands` takes an image as input, and returns the same image with the hand landmarks drawn on it.

        :param img: The image to process
        :param draw: If True, the landmarks will be drawn on the image, defaults to True (optional)
        :return: The image with the hand landmarks drawn on it.
        """
        imgRGB = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )
        return img

    def findPosition(self, img, handNumber=0, draw=True):
        """
        > It takes an image and a hand number (0 or 1) and returns a list of landmark coordinates

        :param img: the image to draw on
        :param handNumber: which hand to use, defaults to 0 (optional)
        :param draw: If True, draw the landmark on the image, defaults to True (optional)
        :return: The landmark list is being returned.
        """
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNumber]
            for id, lm in enumerate(myHand.landmark):
                imgHeight, imgWidth, imgColor = img.shape
                # # lm => xyz coordinate in decimals | decimals represent the ratio of the img
                # # xy-coordinate | (x * width, y * height) locations on screen.
                cx, cy = int(lm.x * imgWidth), int(lm.y * imgHeight)
                self.lmList.append([id, cx, cy])
                if draw and id == self.chosenLm:
                    cv2.circle(
                        img=img,
                        center=(cx, cy),
                        radius=50,
                        color=(0, 255, 0),
                        thickness=10,
                        lineType=cv2.LINE_AA,
                    )

        return self.lmList

    def fingersUp(self):
        """
        If the tip of the finger is higher than the previous joint, then the finger is up
        :return: The fingersUp function returns a list of 1's and 0's. 1's represent fingers that are up and
        0's represent fingers that are down.
        """
        fingers = []

        # Thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers


CAMERA_USB_LOCATION = 0


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(
        CAMERA_USB_LOCATION
    )  # number is based on usb location / index :/
    chosenLm = 8
    detector = handDetector(chosenLm=chosenLm)

    while True:
        # REMEMBER Mirror video
        success, img = cap.read()

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(
            img=img,
            text=(f"FPS: {int(fps)}"),
            org=(10, 70),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=2,
            color=(0, 255, 0),
            thickness=3,
            lineType=cv2.LINE_AA,
        )

        # img = detector.findHands(img=img)
        img = detector.findHands(img=img, draw=False)
        # lmList = detector.findPosition(img=img)
        lmList = detector.findPosition(img=img)

        # if len(lmList) != 0:
        #     print(lmList[chosenLm])

        cv2.imshow(winname="Video", mat=img)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
# chosenLm => the lm position to track.
