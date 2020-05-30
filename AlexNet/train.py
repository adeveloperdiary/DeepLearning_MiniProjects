import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from AlexNet.model import AlexNetModel
from AlexNet.properties import *
from common.dataset.dataset import ClassificationDataset
import pandas as pd
import matplotlib.pyplot as plt
from common.utils.logging_util import *
from common.utils.training_util import *
from AlexNet.transformation import *
import sys


# sys.path.append('/home/home/Downloads/apex')
# sys.path.append('/home/home/Downloads/apex/apex')

# from apex import amp


# from torch.utils.tensorboard import SummaryWriter


# images, ids, _ = next(iter(val_data_loader))
# print(ids.shape, images.shape)

# image = images[0]
# image = torch.Tensor.cpu(image).detach().numpy()
# plt.imsave(f'{INPUT_DIR}/test.jpg', image)

# init_logging()
# create_checkpoint_folder()

def getDataLoader(csv_path, images_path, transformation, fields, training=False, batch_size=16, shuffle=False, num_workers=4, pin_memory=False):
    df = pd.read_csv(csv_path)
    dataset = ClassificationDataset(images_path, df, transformation, fields, training)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return data_loader


fields = {'image': 'image', 'label': 'class'}
train_data_loader = getDataLoader(csv_path=TRAIN_CSV, images_path=TRAIN_DIR, transformation=train_transformation, fields=fields, training=True,
                                  batch_size=512, shuffle=True, num_workers=16, pin_memory=True)
val_data_loader = getDataLoader(csv_path=VALID_CSV, images_path=VALID_DIR, transformation=test_transformation, fields=fields, training=False,
                                batch_size=16, shuffle=True, num_workers=4, pin_memory=True)

# writer = SummaryWriter('runs/alexnet')

# images, ids, _ = next(iter(train_data_loader))
# img_grid = torchvision.utils.make_grid(images[0:12])
# writer.add_image('train_images', img_grid)

model = AlexNetModel(num_classes=256)
# if torch.cuda.device_count() > 1:
#    print(f"Let's use {int(torch.cuda.device_count())} GPUs!")
#    model = torch.nn.DataParallel(model.model)

model.to(DEVICE)
# opt_level = 'O1'
# model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

optimizer = torch.optim.SGD(model.model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005, nesterov=True)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.0005)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

criterion = torch.nn.CrossEntropyLoss()
model.cuda()
log_interval = 1
loss_hist = Averager()

# writer.add_graph(model, images)
# writer.close()

for epoch in range(100):
    loss_hist.reset()
    model.train()
    for i, (images, labels, image_id) in enumerate(train_data_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(images)
        loss = criterion(outputs, labels.squeeze())

        loss_hist.send(loss.item())

        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #    scaled_loss.backward()

        loss.backward()
        optimizer.step()

    if epoch % log_interval == 0:  # print every 2000 mini-batches
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels, image_id in val_data_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = (100 * correct / total)
        print(f"Epoch: #{epoch} Iteration: #{i} Average loss: {loss_hist.value} Validation Accuracy: {accuracy}")
    scheduler.step(accuracy)
    # scheduler.step(accuracy)
