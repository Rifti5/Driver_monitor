import time


class SleepDetector:
    def __init__(self, alarm_delay: float):
        self.alarm_delay = alarm_delay
        self.alarm = False

        self.__countdown = False
        self.__start_time = 0.0

    def detect(self, eye_data: tuple):
        if all(eye_data):
            if not self.__countdown:
                self.__start_time = time.time()
            self.__countdown = True
        else:
            self.__countdown = False
            self.alarm = False

        if self.__countdown and (time.time() - self.__start_time) >= self.alarm_delay:
            self.alarm = True

    def get_countdown(self):
        if self.__countdown:
            return round(time.time() - self.__start_time, 2)
        else:
            return 0.0
