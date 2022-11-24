import csv
import os

from datetime import datetime


class Logger:
    def __init__(self, path: str):
        current_time = datetime.now()
        self.__date = current_time.strftime('%d-%m-%y')
        self.__path = path
        self.__file = f'{self.__date}.csv'

    def create_log_file(self):
        if not os.path.exists(self.__file):
            with open(f'{self.__path}{self.__file}', 'w') as file:
                pass

    def write_log(self, text=''):
        current_time = datetime.now()
        time = current_time.strftime("%H:%M:%S:%f")[:12]
        log = [time, text]

        with open(f'{self.__path}{self.__file}', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(log)
