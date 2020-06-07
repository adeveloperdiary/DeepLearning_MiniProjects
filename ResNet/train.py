from torch.utils.data import DataLoader
from common.dataset.dataset import ClassificationDataset
import pandas as pd
from ResNet.transformation import *
import timeit
from common.utils.base_executor import *
from ResNet.executor import *
from ResNet.properties import *

if __name__ == '__main__':
    def getDataLoader(csv_path, images_path, transformation, fields, training=False, batch_size=16, shuffle=False, num_workers=4,
                      pin_memory=False,
                      drop_last=True):
        df = pd.read_csv(csv_path)
        dataset = ClassificationDataset(images_path, df, transformation, fields, training, mean_rgb=f"{config['INPUT_DIR']}/rgb_val.json")
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
        return data_loader


    fields = {'image': 'image', 'label': 'class'}
    train_data_loader = getDataLoader(csv_path=config['TRAIN_CSV'], images_path=config['TRAIN_DIR'], transformation=train_transformation,
                                      fields=fields,
                                      training=True,
                                      batch_size=128, shuffle=True, num_workers=16, pin_memory=True)
    val_data_loader = getDataLoader(csv_path=config['VALID_CSV'], images_path=config['VALID_DIR'], transformation=test_transformation, fields=fields,
                                    training=False,
                                    batch_size=64, shuffle=True, num_workers=8, pin_memory=True, drop_last=False)

    e = Executor("", {'TRAIN': train_data_loader, 'VAL': val_data_loader}, config=config)

    start = timeit.default_timer()
    e.train()
    stop = timeit.default_timer()
    print(f'Training Time: {round((stop - start) / 60, 2)} Minutes')
