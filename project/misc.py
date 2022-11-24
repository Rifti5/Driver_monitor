import numpy
import cv2
import math
import playsound
import time

def set_frame_writable(frame: numpy.array, is_writable: bool):
    frame.flags.writeable = is_writable


def convert_color_bgr_to_rgb(frame: numpy.array):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def convert_color_rgb_to_bgr(frame: numpy.array):
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def get_euclidian_distance(point1: list, point2: list):
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]

    return abs(math.sqrt((x1-x2)**2 + (y1 - y2)**2))


def vec3_rad_to_deg(rotation_vector):
    rad = 180 / math.pi

    roll = float(rotation_vector[2] * rad)
    yaw = float(rotation_vector[1] * rad)
    pitch = float(rotation_vector[0] * rad)

    return roll, pitch, yaw


def draw_text(image: numpy.array, text: str, org: tuple, color: tuple = (255, 255, 255), scale: int = 0.5, thickness: int = 1):
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def play_alarm():
    playsound.playsound('alarm.wav', block=True)
