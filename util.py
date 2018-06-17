import matplotlib.image as mpimg
import os
import numpy

import constansts

def one_hot_to_num(lbl):
    return numpy.argmax(lbl, axis=1)

def channel_first(tensor):
    return numpy.rollaxis(tensor, 3, 1)

'''
TRAIN DATA
'''


def load_train_data(tiling=True):
    data_dir = 'data/training/'
    train_data_filename = data_dir + 'images/'
    train_labels_filename = data_dir + 'groundtruth/'

    # Extract it into numpy arrays.
    train_data = _extract_data(train_data_filename, constansts.N_SAMPLES, tiling)
    train_labels = _extract_labels(train_labels_filename, constansts.N_SAMPLES, tiling)
    return train_data, train_labels


'''
TEST DATA
'''


def load_test_data(tiling=True):
    data_dir = 'data/test_images/'
    imgs = []
    for filename in os.listdir(data_dir):
        path = data_dir + filename
        if os.path.isfile(path):
            print('Loading ' + path)
            img = mpimg.imread(path)
            imgs.append(img)
        else:
            print('File ' + path + ' does not exist')

    if tiling:
        imgs = _cut_tiles_img(imgs)

    return numpy.asarray(imgs)


'''
PRIVATE METHODS
'''


def _extract_data(filename, num_images, tiling=True):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')

    if tiling:
        imgs = _cut_tiles_img(imgs)

    return numpy.asarray(imgs)


# Extract label images
def _extract_labels(filename, num_images, tiling=True):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')

    if tiling:
        data = _cut_tiles_lbl(gt_imgs)
        labels = numpy.asarray([_value_to_class(numpy.mean(data[i])) for i in range(len(data))])
    else:
        labels = numpy.asarray(gt_imgs)

    return labels.astype(numpy.float32)


# Extract patches from a given image
def _img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j:j + w, i:i + h]
            else:
                im_patch = im[j:j + w, i:i + h, :]
            list_patches.append(im_patch)
    return list_patches


# Assign a label to a patch v
def _value_to_class(v):
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = numpy.sum(v)
    if df > foreground_threshold:
        return [0, 1]
    else:
        return [1, 0]


def _cut_tiles_img(imgs):
    num_images = len(imgs)
    img_patches = [_img_crop(imgs[i], constansts.IMG_PATCH_SIZE, constansts.IMG_PATCH_SIZE) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]
    return data


def _cut_tiles_lbl(gt_imgs):
    num_images = len(gt_imgs)
    gt_patches = [_img_crop(gt_imgs[i], constansts.IMG_PATCH_SIZE, constansts.IMG_PATCH_SIZE) for i in
                  range(num_images)]
    data = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    return data