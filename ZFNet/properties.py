import os
import torch

config = dict()

config['PROJECT_NAME'] = 'zfnet'
config['INPUT_DIR'] = '/media/4TB/datasets/caltech/processed'

config['TRAIN_DIR'] = f"{config['INPUT_DIR']}/train"
config['VALID_DIR'] = f"{config['INPUT_DIR']}/val"

config['TRAIN_CSV'] = f"{config['INPUT_DIR']}/train.csv"
config['VALID_CSV'] = f"{config['INPUT_DIR']}/val.csv"

config['CHECKPOINT_INTERVAL'] = 12
config['NUM_CLASSES'] = 256
config['EPOCHS'] = 10

# ======================================= DEFAULT ============================================= #

config['DEVICE'] = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
config['MULTI_GPU'] = False
config['FP16_MIXED'] = False

config["LOGFILE"] = "output.log"
config["LOGLEVEL"] = "INFO"
