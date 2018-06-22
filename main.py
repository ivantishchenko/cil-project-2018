from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import constants
import tensorflow as tf
import util

tf.logging.set_verbosity(tf.logging.INFO)

weight_decay = 1e-4
growth_rate = 12
depth = 100
compression = 0.5


def cnn_model_fn(features, labels, mode):
    """
    Model function for CNN.
    :param features: input X fed to the estimator
    :param labels: input Y fed to the estimator
    :param mode: TRAIN, EVAL, PREDICT
    :return: tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    """

    # Input Layer
    # 4-D tensor: [batch_size, width, height, channels]
    input_layer = features["x"]

    # Image augmentation
    if mode == tf.estimator.ModeKeys.TRAIN:
        # FLIP UP DOWN
        flip_ud = lambda x: tf.image.random_flip_up_down(x)
        input_layer = tf.map_fn(flip_ud, input_layer)

        # FLIP LEFT RIGHT
        flip_lr = lambda x: tf.image.flip_left_right(x)
        input_layer = tf.map_fn(flip_lr, input_layer)

        # BRIGHTNESS
        # bright = lambda x: tf.image.random_brightness(x, max_delta=0.00005)
        # input_layer = tf.map_fn(bright, input_layer)

        # CONTRAST
        contrast = lambda x: tf.image.random_contrast(x, lower=0.7, upper=1.1)
        input_layer = tf.map_fn(contrast, input_layer)

        # HUE
        # hue = lambda x: tf.image.random_hue(x, max_delta=0.1)
        # input_layer = tf.map_fn(hue, input_layer)

        # # SATURATION
        # satur = lambda x: tf.image.random_saturation(x, lower=0.1, upper=0.15)
        # input_layer = tf.map_fn(satur, input_layer)

        tf.summary.image('Augmentation', input_layer, max_outputs=16)

    # DenseNet
    n_blocks = (depth - 4) // 6
    n_channels = growth_rate * 2

    def dense_layer(x):
        return tf.layers.dense(x, units=2,
                               activation=tf.nn.softmax,
                               kernel_initializer=tf.keras.initializers.he_normal(),
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

    def bottleneck(x):
        channels = growth_rate * 4
        x = bn_relu(x)
        x = conv(x, channels, (1, 1))
        x = bn_relu(x)
        x = conv(x, growth_rate, (3, 3))
        return x

    def transition(x, in_channels):
        out_channels = int(in_channels * compression)
        x = bn_relu(x)
        x = conv(x, out_channels, (1, 1))
        x = tf.layers.average_pooling2d(x, (2, 2), strides=(2, 2))
        return x, out_channels

    def dense_block(x, blocks, n_channels):
        concat = x
        for i in range(blocks):
            x = bottleneck(concat)
            concat = tf.concat([x, concat], axis=-1)
            n_channels += growth_rate
        return concat, n_channels

    def bn_relu(x):
        x = tf.layers.batch_normalization(x, momentum=0.9, epsilon=1e-5)
        x = tf.nn.relu(x)
        return x

    def conv(x, out_filters, k_size):
        return tf.layers.conv2d(x, filters=out_filters,
                                kernel_size=k_size,
                                strides=(1, 1),
                                padding='same',
                                kernel_initializer=tf.keras.initializers.he_normal(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                use_bias=False)

    x = conv(input_layer, n_channels, (3, 3))
    x, n_channels = dense_block(x, n_blocks, n_channels)
    x, n_channels = transition(x, n_channels)
    x, n_channels = dense_block(x, n_blocks, n_channels)
    x, n_channels = transition(x, n_channels)
    x, n_channels = dense_block(x, n_blocks, n_channels)
    x = bn_relu(x)
    x = tf.layers.average_pooling2d(x, pool_size=(x.shape[1], x.shape[2]), strides=(1, 1), padding='valid')

    x_shape = x.get_shape().as_list()
    x_flat = tf.reshape(x, [-1, x_shape[1] * x_shape[2] * x_shape[3]])
    logits = dense_layer(x_flat)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": logits
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # load provided images
    train_data = util.load_train_img(tiling=False)
    train_labels = util.load_train_lbl(tiling=True)
    predict_data = util.load_test_data(tiling=False)
    train_labels = util.one_hot_to_num(train_labels)
    # expansion
    train_data = util.crete_patches_large(train_data, constants.IMG_PATCH_SIZE, 16, constants.PADDING, is_mask=False)
    predict_data = util.crete_patches_large(predict_data, constants.IMG_PATCH_SIZE, 16, constants.PADDING,
                                            is_mask=False)

    # Create the Estimator
    road_estimator = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="outputs/road")

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=constants.BATCH_SIZE,
        num_epochs=None,
        shuffle=True)

    road_estimator.train(
        input_fn=train_input_fn,
        max_steps=(constants.N_SAMPLES * constants.NUM_EPOCH) // constants.BATCH_SIZE)

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        num_epochs=1,
        shuffle=False)

    eval_results = road_estimator.evaluate(input_fn=eval_input_fn)
    print(eval_results)

    # Do prediction on test data
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": predict_data},
        num_epochs=1,
        shuffle=False)

    predictions = road_estimator.predict(input_fn=predict_input_fn)
    res = [p['probabilities'] for p in predictions]

    file_names = util.get_file_names()
    util.create_prediction_dir("predictions_test/")
    offset = 1444

    for i in range(1, constants.N_TEST_SAMPLES + 1):
        img = util.label_to_img_inverse(608, 608, 16, 16, res[(i - 1) * offset:i * offset])
        img = util.img_float_to_uint8(img)
        Image.fromarray(img).save('predictions_test/' + file_names[i - 1])

    # Predictions Train
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        num_epochs=1,
        shuffle=False)

    predictions = road_estimator.predict(input_fn=predict_input_fn)

    res = [p['probabilities'] for p in predictions]
    util.create_prediction_dir("predictions_train/")
    for i in range(1, 101):
        img = util.label_to_img_inverse(400, 400, constants.IMG_PATCH_SIZE, constants.IMG_PATCH_SIZE,
                                        res[(i - 1) * 625:i * 625])
        img = util.img_float_to_uint8(img)
        Image.fromarray(img).save('predictions_train/{:03}.png'.format(i))


if __name__ == "__main__":
    tf.app.run()
