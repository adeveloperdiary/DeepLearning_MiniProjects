import os
import sys

sys.path.append('/media/home/1TB_Disk/projects/Image_Captions/coco_api/cocoapi/PythonAPI')
sys.path.append('Image_Caption_using_CNN-RNN')
from pycocotools.coco import COCO

# initialize COCO API for instance annotations
dataDir = '/media/4TB/datasets/coco_2014'
dataType = 'train2014'
instances_annFile = os.path.join(dataDir, 'annotations/instances_{}.json'.format(dataType))
coco = COCO(instances_annFile)

# initialize COCO API for caption annotations
captions_annFile = os.path.join(dataDir, 'annotations/captions_{}.json'.format(dataType))
coco_caps = COCO(captions_annFile)

# get image ids
ids = list(coco.anns.keys())

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

# pick a random image and obtain the corresponding URL
ann_id = np.random.choice(ids)
img_id = coco.anns[ann_id]['image_id']
print(img_id)
img = coco.loadImgs(img_id)[0]
url = img['coco_url']

# print URL and visualize corresponding image
print(url)
I = io.imread(url)
plt.axis('off')
plt.imshow(I)
plt.show()

print(coco.loadImgs(img_id))
# load and display captions
annIds = coco_caps.getAnnIds(imgIds=img['id'])
anns = coco_caps.loadAnns(annIds)
coco_caps.showAnns(anns)
