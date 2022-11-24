import cv2


class Camera:
    def __init__(self, number: int):
        self.cap = cv2.VideoCapture(number)

    def is_opened(self):
        return self.cap.isOpened()

    def get_frame(self):
        success, frame = self.cap.read()
        return success, frame

    def get_framerate(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    def release(self):
        self.cap.release()

    def get_frame_shape(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

