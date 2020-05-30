import datetime
import os
import logging


def create_checkpoint_folder():
    global CHECKPOINT_PATH

    DIR_INPUT = os.environ.get("DIR_INPUT", "/temp/")

    CHECKPOINT_PATH = f'{DIR_INPUT}/checkpoint/{datetime.datetime.now().strftime("%b-%d-%Y-%H-%M-%S")}'
    logging.info(f"Creating checkpoint folder ...{CHECKPOINT_PATH}")
    os.mkdir(CHECKPOINT_PATH)


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0
