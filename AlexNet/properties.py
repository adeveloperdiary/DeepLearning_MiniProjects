import os
import torch

INPUT_DIR = '/media/4TB/datasets/caltech/processed'
TRAIN_DIR = f'{INPUT_DIR}/train'
VALID_DIR = f'{INPUT_DIR}/val'

TRAIN_CSV = f'{INPUT_DIR}/train.csv'
VALID_CSV = f'{INPUT_DIR}/val.csv'

os.environ["LOGFILE"] = "output.log"
os.environ["LOGLEVEL"] = "INFO"

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

CHECKPOINT_PATH = ""
