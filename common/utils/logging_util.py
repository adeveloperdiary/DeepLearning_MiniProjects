import os
import logging
import logging.handlers


def init_logging():
    handler = logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE", "output.log"))
    formatter = logging.Formatter(logging.BASIC_FORMAT)
    handler.setFormatter(formatter)
    training = logging.getLogger()
    training.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    training.addHandler(handler)
