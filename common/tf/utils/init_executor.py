import os
import logging
import logging.handlers
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class InitExecutor(object):
    def __init__(self):
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_loss_hist = None
        self.val_loss_hist = None
        self.pbar = None
        self.last_checkpoint_file = None
        self.load_from_check_point = None
        self.logger = None
        self.tracking = None
        self.tb_writer = None
        self.val_acc = None
        self.train_acc = None
        self.train_loss = None
        self.val_loss = None
        self.learning_rate = None

        self.CHECKPOINT_PATH = None
        self.CHECKPOINT_INTERVAL = None
        self.NUM_CLASSES = None
        self.INPUT_DIR = None
        self.TRAIN_DIR = None
        self.VALID_DIR = None
        self.TRAIN_CSV = None
        self.VALID_CSV = None
        self.DEVICE = None
        self.EPOCHS = None
        self.PROJECT_NAME = None
        self.LOGFILE = None
        self.LOGLEVEL = None
        self.MULTI_GPU = None
        self.FP16_MIXED = None

        self.TRAIN_DATA_SIZE = None
        self.TRAIN_BATCH_SIZE = None
        self.VAL_DATA_SIZE = None
        self.VAL_BATCH_SIZE = None

    def init_logging(self):
        """
            Initialize the logger so that both console and file logging can be enabled
        """

        # Initialize logging
        log = logging.getLogger()
        log.setLevel(self.LOGLEVEL)
        formatter = logging.Formatter(logging.BASIC_FORMAT)

        # Create file handler
        file_handler = logging.handlers.WatchedFileHandler(self.LOGFILE)
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)

        # Initialize the logger instance
        self.logger = BaseLogger(log)

    def init_checkpoint(self):
        """
            The following logic is to automatically determine whether to load from
            the last checkpoint and determine the last checkpoint file.
        """

        # If the last.checkpoint file exists in the input dir
        if os.path.isfile(f'{self.INPUT_DIR}/last.checkpoint.{self.PROJECT_NAME}'):
            # Open the file and read the first line
            with open(f'{self.INPUT_DIR}/last.checkpoint.{self.PROJECT_NAME}', 'r') as file:
                lines = file.readlines()
                self.last_checkpoint_file = lines[0].strip()
                # If the file exists then load from last checkpoint
                if os.path.isfile(self.last_checkpoint_file):
                    self.load_from_check_point = True
                    # Also load the tb writer from previous session
                    self.tb_writer = SummaryWriter(log_dir=lines[1])
        else:
            # Initialize the tensor board summary writer
            now = datetime.now()
            self.tb_writer = SummaryWriter(log_dir=f'runs/{self.PROJECT_NAME}_{now.strftime("%Y%m%d-%H%M%S")}')


class BaseLogger:
    def __init__(self, logger):
        self.logger = logger

    def info(self, message):
        self.logger.info(message)
        # print(message)

    def error(self, message):
        self.logger.error(message)
        # print(message)

    def warning(self, message):
        self.logger.warning(message)
        # print(message)
