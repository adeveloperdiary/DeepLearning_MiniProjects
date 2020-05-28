import datetime
import os
import logging


def create_checkpoint_folder():
    global CHECKPOINT_PATH

    DIR_INPUT = os.environ.get("DIR_INPUT", "/temp/")

    CHECKPOINT_PATH = f'{DIR_INPUT}/checkpoint/{datetime.datetime.now().strftime("%b-%d-%Y-%H-%M-%S")}'
    logging.info(f"Creating checkpoint folder ...{CHECKPOINT_PATH}")
    os.mkdir(CHECKPOINT_PATH)
