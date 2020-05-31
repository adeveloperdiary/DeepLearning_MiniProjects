import os
import torch

config = dict()

config['INPUT_DIR'] = '/media/4TB/datasets/caltech/processed'
config['TRAIN_DIR'] = f"{config['INPUT_DIR']}/train"
config['VALID_DIR'] = f"{config['INPUT_DIR']}/val"

config['TRAIN_CSV'] = f"{config['INPUT_DIR']}/train.csv"
config['VALID_CSV'] = f"{config['INPUT_DIR']}/val.csv"




config['NUM_CLASSES'] = 256
config['EPOCHS'] = 50

config['DEVICE'] = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

os.environ["LOGFILE"] = "output.log"
os.environ["LOGLEVEL"] = "INFO"
