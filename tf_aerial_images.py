
import gzip
import os
import sys
import urllib
import matplotlib as plt
import matplotlib.image as mpimg
import matplotlib.colors as mpcol
import imgaug as ia
from imgaug import augmenters as iaa
from PIL import Image
from os import listdir, path
from parse import parse
from skimage.restoration import denoise_nl_means
from shutil import copyfile
from math import floor, isnan, ceil

import datetime

import code

import tensorflow.python.platform

import numpy
import tensorflow as tf


NUM_CHANNELS = 3 # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 100
TEST_SIZE = 94
SEED = 1  # Set to None for random seed.
TRAINING_BATCH_SIZE = 32
PREDICTION_BATCH_SIZE = 512
NUM_EPOCHS = 40
RESTORE_MODEL = False # If True, restore existing model instead of training a new one
RECORDING_STEP = 1000


# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16
IMG_FRAME_SIZE = 16

if len(sys.argv) != 2:
    print("usage: $0 logdir")
    sys.exit()

tf.app.flags.DEFINE_string('train_dir', sys.argv[1],
                           """Directory where to write event logs """
                           """and checkpoint.""")
FLAGS = tf.app.flags.FLAGS

# Extract patches from a given image
def img_crop(im, w, h, frame, oversize):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]

    img_ul_corner = (0, 0)
    offset = 0

    if oversize:
        offset = frame
        imgwidth = int(floor(imgwidth / 3))
        imgheight = int(floor(imgheight / 3))
        img_ul_corner = (imgheight, imgwidth)

    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[(img_ul_corner[0]-offset+j):(img_ul_corner[0]+j+w+offset), (img_ul_corner[1]-offset+i):(img_ul_corner[1]+i+h+offset)]
            else:
                im_patch = im[(img_ul_corner[0]-offset+j):(img_ul_corner[0]+j+w+offset), (img_ul_corner[1]-offset+i):(img_ul_corner[1]+i+h+offset), :]

            list_patches.append(im_patch)
    return list_patches

def contrast_stretch(img):

    res = numpy.zeros(img.shape, dtype=int)
    byteimg = (255*img).astype(int)

    for i in range(img.shape[2]):
        converted_channel = numpy.zeros((img.shape[0], img.shape[1]), dtype=int)
        h = numpy.histogram(byteimg[:,:,i], bins=range(256))
        cdfmin = h[0][0]
        cdf = 0        

        for j in range(len(h[0])):
            cdf = cdf + h[0][j]
            ids = numpy.where(img[:,:,i] == h[1][j])
            converted_channel[tuple(ids)] = (cdf-cdfmin)/(img.shape[0]*img.shape[1]-cdfmin) * 255

        res[:,:,i] = converted_channel
            
    return res.astype(numpy.float32) / 255

def standardize(img):
    img -= img.mean()
    img /= img.std()

    return img

def mirror_and_concat_img(img):

    ul = numpy.fliplr(numpy.flipud(img))
    ur = numpy.fliplr(numpy.flipud(img))
    u = numpy.flipud(img)
    l = numpy.fliplr(img)
    r = numpy.fliplr(img)
    b = numpy.flipud(img)
    br = numpy.fliplr(numpy.flipud(img))
    bl = numpy.fliplr(numpy.flipud(img))

    top_row = numpy.concatenate((ul, u, ur), axis=1)
    middle_row = numpy.concatenate((l, img, r), axis=1)
    bottom_row = numpy.concatenate((bl, b, br), axis=1)

    out = numpy.concatenate((top_row, middle_row, bottom_row), axis=0)

    return out


def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)

    img_patches = [img_crop(mirror_and_concat_img(imgs[i]), IMG_PATCH_SIZE, IMG_PATCH_SIZE, IMG_FRAME_SIZE, True) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    return numpy.asarray(data)
        
# Assign a label to a patch v
def value_to_class(v):
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = numpy.sum(v)
    if df > foreground_threshold:
        return [1, 0]
    else:
        return [0, 1]

# Extract label images
def extract_labels(filename, num_images):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE, IMG_FRAME_SIZE, False) for i in range(num_images)]
    data = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = numpy.asarray([value_to_class(numpy.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(numpy.float32)


def augment_training_data(data, labels):
    # img = standardize(img)
    # img = contrast_stretch(img)
    # img = denoise_nl_means(img, patch_size=2, patch_distance=2, multichannel=True)

    ia.seed(SEED)
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential(
        [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),

            sometimes(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode="edge",
                pad_cval=(0, 255)
            )),
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-45, 45),
                shear=(-16, 16),
                order=[0, 1],
                cval=(0, 255),
                mode="symmetric"
            )),

            iaa.SomeOf((0, 2),
                       [
                           sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),

                           iaa.Sharpen(alpha=(0.9, 1.0), lightness=(0.75, 1.25)),
                           iaa.Emboss(alpha=(0.9, 1.0), strength=(0, 0.25)),

                           iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),

                           iaa.Dropout((0.01, 0.05), per_channel=0.5),

                           iaa.Add((-10, 10), per_channel=0.5),

                           iaa.AddToHueAndSaturation((-5, 5)),

                           iaa.ContrastNormalization((0.8, 1.1), per_channel=0.5),
                           sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),

                           sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),

                           sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                       ],
                       random_order=True
                       )
        ],
        random_order=True
    )

    data8 = numpy.uint8(data*255.0)
    augmented_data8 = numpy.empty((0, data8.shape[1], data8.shape[2], data8.shape[3]))
    augmented_labels = numpy.empty((0, 2))

    for s in range(10):
        augmented_set = seq.augment_images(data8)
        augmented_data8 = numpy.concatenate((augmented_data8, augmented_set), axis=0)
        augmented_labels = numpy.concatenate((augmented_labels, labels), axis=0)


    augmented_data = numpy.float32(augmented_data8) / 255.0

    return augmented_data, augmented_labels


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0 *
        numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) /
        predictions.shape[0])

# Write predictions from neural network to a file
def write_predictions_to_file(predictions, labels, filename):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    file = open(filename, "w")
    n = predictions.shape[0]
    for i in range(0, n):
        file.write(max_labels(i) + ' ' + max_predictions(i))
    file.close()

# Print predictions from neural network
def print_predictions(predictions, labels):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    print (str(max_labels) + ' ' + str(max_predictions))

# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    array_labels = numpy.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if labels[idx][0] > 0.5:
                l = 1
            else:
                l = 0
            array_labels[j:j+w, i:i+h] = l
            idx = idx + 1
    return array_labels

def img_float_to_uint8(img):
    rimg = img - numpy.min(img)
    rimg = (rimg / numpy.max(rimg) * PIXEL_DEPTH).round().astype(numpy.uint8)
    return rimg

def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = numpy.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = numpy.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    color_mask[:,:,0] = predicted_img*PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def make_img(predicted_img):
    img8 = img_float_to_uint8(predicted_img)
    image = Image.fromarray(img8, 'L').convert("RGBA")
    return image

def save_images(imgs, output_dir):

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    cntr = 0
    for img in imgs:
        im = Image.fromarray(numpy.uint8(img * 255.0))
        im.save(output_dir + str("/") + str(cntr) + ".png")
        cntr += 1

def main(argv=None):  # pylint: disable=unused-argument

    timestamp = "{}".format(datetime.datetime.now().strftime("%d-%B-%H:%M:%S"))
    data_dir = 'training/'
    test_dir = 'test_images/'
    train_data_path = data_dir + 'images/'
    train_labels_path = data_dir + 'groundtruth/' 

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_path, TRAINING_SIZE)
    train_labels = extract_labels(train_labels_path, TRAINING_SIZE)
    train_data, train_labels = augment_training_data(train_data, train_labels)

    print("Patch # after augmentation: ", train_data.shape[0])

    #save_images(train_data, "augmented_training_data")
    #sys.exit()

    num_epochs = NUM_EPOCHS

    c0 = 0
    c1 = 0
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    print('Balancing training data...')
    min_c = min(c0, c1)
    idx0 = [i for i, j in enumerate(train_labels) if j[0] == 1]
    idx1 = [i for i, j in enumerate(train_labels) if j[1] == 1]
    new_indices = idx0[0:min_c] + idx1[0:min_c]

    train_data = train_data[new_indices,:,:,:]
    train_labels = train_labels[new_indices]


    train_size = train_labels.shape[0]

    c0 = 0
    c1 = 0
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))


    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(
        tf.float32,
        shape=(TRAINING_BATCH_SIZE, IMG_PATCH_SIZE+2*IMG_FRAME_SIZE, IMG_PATCH_SIZE+2*IMG_FRAME_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.float32,
                                       shape=(TRAINING_BATCH_SIZE, NUM_LABELS))
    #train_all_data_node = tf.constant(train_data)

    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when when we call:
    # {tf.initialize_all_variables().run()}
    conv1_weights = tf.Variable(
        tf.truncated_normal([3, 3, NUM_CHANNELS, 32],
                            stddev=0.1,
                            seed=SEED))
    conv1_biases = tf.Variable(tf.zeros([32]))
    conv2_weights = tf.Variable(
        tf.truncated_normal([3, 3, 32, 64],
                            stddev=0.1,
                            seed=SEED))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))

    conv3_weights = tf.Variable(
        tf.truncated_normal([3, 3, 64, 128],
                            stddev=0.1,
                            seed=SEED))
    conv3_biases = tf.Variable(tf.constant(0.1, shape=[128]))

    conv4_weights = tf.Variable(
        tf.truncated_normal([3, 3, 128, 256],
                            stddev=0.1,
                            seed=SEED))
    conv4_biases = tf.Variable(tf.constant(0.1, shape=[256]))

    conv5_weights = tf.Variable(
        tf.truncated_normal([3, 3, 256, 256],
                            stddev=0.1,
                            seed=SEED))
    conv5_biases = tf.Variable(tf.constant(0.1, shape=[256]))

    fc1_weights = tf.Variable(
        tf.truncated_normal([int((IMG_PATCH_SIZE+IMG_FRAME_SIZE) / 4 * (IMG_PATCH_SIZE+IMG_FRAME_SIZE) / 8 * 32), 512],
                            stddev=0.1,
                            seed=SEED))

    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
    fc2_weights = tf.Variable(
        tf.truncated_normal([512, NUM_LABELS],
                            stddev=0.1,
                            seed=SEED))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

    # Make an image summary for 4d tensor image with index idx
    def get_image_summary(img, idx = 0):
        V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
        img_w = img.get_shape().as_list()[1]
        img_h = img.get_shape().as_list()[2]
        min_value = tf.reduce_min(V)
        V = V - min_value
        max_value = tf.reduce_max(V)
        V = V / (max_value*PIXEL_DEPTH)
        V = tf.reshape(V, (img_w, img_h, 1))
        V = tf.transpose(V, (2, 0, 1))
        V = tf.reshape(V, (-1, img_w, img_h, 1))
        return V
    
    # Make an image summary for 3d tensor image with index idx
    def get_image_summary_3d(img):
        V = tf.slice(img, (0, 0, 0), (1, -1, -1))
        img_w = img.get_shape().as_list()[1]
        img_h = img.get_shape().as_list()[2]
        V = tf.reshape(V, (img_w, img_h, 1))
        V = tf.transpose(V, (2, 0, 1))
        V = tf.reshape(V, (-1, img_w, img_h, 1))
        return V

    # Get prediction for given input image 
    def get_prediction(img):

        data_node = tf.placeholder(tf.float32, shape=(PREDICTION_BATCH_SIZE, IMG_PATCH_SIZE + 2 * IMG_FRAME_SIZE, IMG_PATCH_SIZE + 2 * IMG_FRAME_SIZE, NUM_CHANNELS))
        data = numpy.asarray(img_crop(mirror_and_concat_img(img), IMG_PATCH_SIZE, IMG_PATCH_SIZE, IMG_FRAME_SIZE, True))

        output_prediction = numpy.empty((data.shape[0],2))
        indices = range(data.shape[0])

        for k in range(int(ceil(data.shape[0] / PREDICTION_BATCH_SIZE))):

            offs = (k * PREDICTION_BATCH_SIZE)
            batch_ids = indices[offs:(offs+PREDICTION_BATCH_SIZE)]

            n_unclassified = len(batch_ids)

            if not n_unclassified == PREDICTION_BATCH_SIZE:
                overf_ids = indices[0:(PREDICTION_BATCH_SIZE - n_unclassified)]
                batch_ids = numpy.concatenate((batch_ids, overf_ids))

            patches = data[batch_ids,:,:,:]

            fd = {data_node: patches}
            output = tf.nn.softmax(model(data_node))
            res = s.run(output, feed_dict=fd)

            output_prediction[offs:(offs + len(batch_ids)),:] = res[range(n_unclassified)]

        img_prediction = label_to_img(img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)

        return img_prediction

    # Get a concatenation of the prediction and groundtruth for given input file
    def get_prediction_with_groundtruth(img):

        img_prediction = get_prediction(img)
        cimg = concatenate_images(img, img_prediction)

        return cimg

    # Get prediction overlaid on the original image for given input file
    def get_prediction_with_overlay(img):

        img_prediction = get_prediction(img)
        oimg = make_img_overlay(img, img_prediction)

        return oimg

    def classify_files_in_dir(input_dir, output_dir, count):
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        file_names = [f for f in sorted(listdir(input_dir)) if path.isfile(path.join(input_dir, f))]
        for i in range(count):
            filename = file_names[i]

            img = mpimg.imread(input_dir + filename)

            pimg = get_prediction_with_groundtruth(img)
            Image.fromarray(pimg).save(output_dir + "prediction_" + filename)
            oimg = get_prediction_with_overlay(img)
            oimg.save(output_dir + "overlay_" + str(filename))
            pimg = get_prediction(img)
            pimg = make_img(pimg)
            pimg.save(output_dir + "mask_" + str(filename))

    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    def model(data, train=False):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].

        conv = tf.nn.conv2d(data,
                            conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        conv2 = tf.nn.conv2d(pool,
                            conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        pool2 = tf.nn.max_pool(relu2,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        conv3 = tf.nn.conv2d(pool2,
                            conv3_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
        pool3 = tf.nn.max_pool(relu3,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        conv4 = tf.nn.conv2d(pool3,
                            conv4_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))
        pool4 = tf.nn.max_pool(relu4,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        conv5 = tf.nn.conv2d(pool4,
                            conv5_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu5 = tf.nn.relu(tf.nn.bias_add(conv5, conv5_biases))
        pool5 = tf.nn.max_pool(relu5,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')


        #print('data ' + str(data.get_shape()))
        #print('conv ' + str(conv.get_shape()))
        #print('relu ' + str(relu.get_shape()))
        #print('pool ' + str(pool.get_shape()))
        #print('pool2 ' + str(pool2.get_shape()))
        #print('pool3 ' + str(pool3.get_shape()))
        #print('pool4 ' + str(pool4.get_shape()))
        #print('pool5 ' + str(pool5.get_shape()))

        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool5.get_shape().as_list()
        flattened = tf.reshape(
            pool5,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(flattened, fc1_weights) + fc1_biases)
        #hidden = tf.layers.dense(inputs=flattened, units=512, activation=tf.nn.relu)

        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)

        out = tf.matmul(hidden, fc2_weights) + fc2_biases
        #out = tf.layers.dense(inputs=hidden, units=2, activation=tf.nn.relu)

        return out

    # Training computation: logits + cross-entropy loss.
    logits = model(train_data_node, True) # TRAINING_BATCH_SIZE*NUM_LABELS
    # print 'logits = ' + str(logits.get_shape()) + ' train_labels_node = ' + str(train_labels_node.get_shape())
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=train_labels_node))
    tf.summary.scalar('loss', loss)

    all_params_node = [conv1_weights, conv1_biases, conv2_weights, conv2_biases, conv3_weights, conv3_biases, conv4_weights, conv4_biases, conv5_biases, conv5_weights, fc1_biases, fc1_weights]
    all_params_names = ['conv1_weights', 'conv1_biases', 'conv2_weights', 'conv2_biases', 'conv3_weights', 'conv3_biases', 'conv4_weights', 'conv4_biases', 'conv5_biases', 'conv5_weights', 'fc1_biases', 'fc1_weights']
    all_grads_node = tf.gradients(loss, all_params_node)
    all_grad_norms_node = []
    for i in range(0, len(all_grads_node)):
        norm_grad_i = tf.global_norm([all_grads_node[i]])
        all_grad_norms_node.append(norm_grad_i)
        tf.summary.scalar(all_params_names[i], norm_grad_i)
    
    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                    tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    # Add the regularization term to the loss.
    loss += 5e-4 * regularizers

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0)
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        0.01,                # Base learning rate.
        batch * TRAINING_BATCH_SIZE,  # Current index into the dataset.
        train_size,          # Decay step.
        0.98,                # Decay rate.
        staircase=True)

    momentum = tf.train.exponential_decay(
        0.005,
        batch * TRAINING_BATCH_SIZE,
        train_size,
        1.00,
        staircase=True)

    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('momentum', momentum)
    
    # Use simple momentum for the optimization.
    optimizer = tf.train.MomentumOptimizer(learning_rate,
                                           momentum).minimize(loss,
                                                         global_step=batch)

    # Predictions for the minibatch, validation set and test set.
    train_prediction = tf.nn.softmax(logits)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

    # Create a local session to run this computation.
    #with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as s:
    with tf.Session() as s:

        if RESTORE_MODEL:
            # Restore variables from disk.
            saver.restore(s, FLAGS.train_dir + "/model.ckpt")
            print("Model restored.")

        else:
            # Run all the initializers to prepare the trainable parameters.
            tf.global_variables_initializer().run()

            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(FLAGS.train_dir + "/" + timestamp, graph=s.graph)
            print('Initialized!')
            # Loop through training steps.
            print('Total number of iterations = ' + str(int(num_epochs * train_size / TRAINING_BATCH_SIZE)))

            training_indices = range(train_size)

            for iepoch in range(num_epochs):

                # Permute training indices
                perm_indices = numpy.random.permutation(training_indices)

                for step in range (int(ceil(train_size / TRAINING_BATCH_SIZE))):

                    # wtf how is this supposed to work???
                    #offset = (step * TRAINING_BATCH_SIZE) % (train_size - TRAINING_BATCH_SIZE)

                    offset = (step * TRAINING_BATCH_SIZE)
                    batch_indices = perm_indices[offset:(offset + TRAINING_BATCH_SIZE)]

                    if not len(batch_indices) == TRAINING_BATCH_SIZE:
                        overflow_ids = perm_indices[0:(TRAINING_BATCH_SIZE-len(batch_indices))]
                        batch_indices = numpy.concatenate((batch_indices, overflow_ids))

                    # Compute the offset of the current minibatch in the data.
                    # Note that we could use better randomization across epochs.
                    batch_data = train_data[batch_indices, :, :, :]
                    batch_labels = train_labels[batch_indices]
                    # This dictionary maps the batch data (as a numpy array) to the
                    # node in the graph is should be fed to.

                    feed_dict = {train_data_node: batch_data,
                                 train_labels_node: batch_labels}

                    if step % RECORDING_STEP == 0:

                        summary_str, _, l, lr, m, predictions = s.run(
                            [summary_op, optimizer, loss, learning_rate, momentum, train_prediction],
                            feed_dict=feed_dict)
                        #summary_str = s.run(summary_op, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.flush()

                        # print_predictions(predictions, batch_labels)

                        print('Epoch %d:%.2f' % (iepoch, float(step) * TRAINING_BATCH_SIZE / train_size))
                        print('Minibatch loss: %.3f, learning rate: %.6f, momentum: %.6f' % (l, lr, m))
                        print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))

                        assert not numpy.isnan(l), 'Model diverged with loss = NaN'

                        sys.stdout.flush()
                    else:
                        # Run the graph and fetch some of the nodes.
                        _, l, lr, m, predictions = s.run(
                            [optimizer, loss, learning_rate, momentum, train_prediction],
                            feed_dict=feed_dict)

                # Save the variables to disk.
                save_path = saver.save(s, FLAGS.train_dir + "/model.ckpt")
                print("Model saved in file: %s" % save_path)


        print ("Running prediction on training set")
        classify_files_in_dir(train_data_path, timestamp + "_predictions_training/", TRAINING_SIZE)

        print ("Running prediction on test set")
        classify_files_in_dir(test_dir, timestamp + "_predictions_test/", TEST_SIZE)

        copyfile(sys.argv[0], timestamp + "_predictions_test/model.py")


if __name__ == '__main__':
    tf.app.run()
