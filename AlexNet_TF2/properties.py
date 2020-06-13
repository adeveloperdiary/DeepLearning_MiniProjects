import os

config = dict()

config['PROJECT_NAME'] = 'alexnet'
config['INPUT_DIR'] = '/media/4TB/datasets/caltech/processed_tfrecords'

config['TRAIN_DIR'] = f"{config['INPUT_DIR']}/train"
config['VALID_DIR'] = f"{config['INPUT_DIR']}/val"

config['TRAIN_CSV'] = f"{config['INPUT_DIR']}/train.csv"
config['VALID_CSV'] = f"{config['INPUT_DIR']}/val.csv"

config['CHECKPOINT_INTERVAL'] = 10
config['NUM_CLASSES'] = 257
config['EPOCHS'] = 100

# ======================================= DEFAULT ============================================= #


config['MULTI_GPU'] = False
config['FP16_MIXED'] = False

config["LOGFILE"] = "output.log"
config["LOGLEVEL"] = "INFO"
