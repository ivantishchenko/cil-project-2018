from imgaug import augmenters as iaa
import imgaug as ia
import numpy as np
import matplotlib.image as mpimg
from PIL import Image
import os
import matplotlib.pyplot as plt

MIN = 0
MAX = 208
IMG_PATH = 'data/training/images/'
MAP_PATH = 'data/training/groundtruth/'
PIXEL_DEPTH = 255


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
    return rimg


ia.seed(33)
for filename in os.listdir(IMG_PATH):
    img = mpimg.imread(IMG_PATH + filename)
    # plt.imshow(img)
    img = img_float_to_uint8(img)
    x, y = np.random.randint(MIN, MAX + 1, 2)

    padding = iaa.Pad(
        px=(y, x, MAX - y, MAX - x),
        pad_mode=["symmetric", "reflect", "wrap"],
        keep_size=False
    )
    flip_horizontal = iaa.Fliplr(0.5)
    flip_vertical = iaa.Flipud(0.5)
    affine = iaa.Affine(
        rotate=(-90, 90),
        shear=(-5,5),
        mode=["symmetric", "reflect", "wrap"]
    )

    # TODO: SALT, DROPOUT, CONTRAST_NORMALIZATION, MULTIPLY ONLY IMG NOT GT

    seq = iaa.Sequential([padding, flip_horizontal, flip_vertical, affine]).to_deterministic()
    image_aug = seq.augment_image(img)
    plt.imshow(image_aug)
    plt.show()
