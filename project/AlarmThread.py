import threading

from misc import play_alarm


class AlarmThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super(AlarmThread, self).__init__(*args, **kwargs)
        self.daemon = True
        self.__pause = threading.Event()
        self.__pause.set()
        self.__running = True

    def run(self):
        while self.__running:
            self.__pause.wait()
            play_alarm()

    def pause(self):
        self.__pause.clear()

    def resume(self):
        self.__pause.set()

    def stop(self):
        self.__running = False

