# Provide the input preprocessing location
INPUT_PATH = '/media/4TB/datasets/caltech/256_ObjectCategories'
# Provide the output location to store the processed images
OUTPUT_PATH = '/media/4TB/datasets/caltech/processed_tfrecords'
# Validation split. Range - [ 0.0 - 1.0 ]
VALIDATION_SPLIT = 0.2
# Output image dimension. ( height,width )
OUTPUT_DIM = (256, 256)
# If RGB mean is needed, set this to True
RGB_MEAN = True
# If this is true, then the images will only be resized while preserving the aspect ratio.
CENTER_CROP = True
# If this is true then the smaller side will be resized to the dimension defined above
SMALLER_SIDE_RESIZE = False
# JPEG Compression Ratio
JPEG_QUALITY = 100


# Function to provide the logic to parse the class labels from the directory.
def read_class_labels(path):
    return path.split('/')[-1].split('.')[-1]
