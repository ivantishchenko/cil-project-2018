from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import constants
import tensorflow as tf
import util

tf.logging.set_verbosity(tf.logging.INFO)

TILING = True  # Use 16x16 tiles (True) or feed in image directly to network (False)


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
        contrast = lambda x: tf.image.random_contrast(x, lower=0.98, upper=1.2)
        input_layer = tf.map_fn(contrast, input_layer)

        # HUE
        # hue = lambda x: tf.image.random_hue(x, max_delta=0.1)
        # input_layer = tf.map_fn(hue, input_layer)

        # # SATURATION
        # satur = lambda x: tf.image.random_saturation(x, lower=0.1, upper=0.15)
        # input_layer = tf.map_fn(satur, input_layer)

        tf.summary.image('Augmentation', input_layer, max_outputs=16)

    # Model
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=256,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=256,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    # flatten
    pool_shape = pool4.get_shape().as_list()
    pool4_flat = tf.reshape(pool4, [-1, pool_shape[1] * pool_shape[2] * pool_shape[3]])

    # FC 2048 neurons
    dense = tf.layers.dense(inputs=pool4_flat, units=2048, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
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
    train_data, train_labels = util.load_train_data(tiling=False)
    predict_data = util.load_test_data(tiling=False)
    # expansion
    train_data = util.crete_patches_large(train_data, constants.IMG_PATCH_SIZE, 16, constants.PADDING, is_mask=False)
    train_labels = util.crete_patches_large(train_labels, constants.IMG_PATCH_SIZE, 16, constants.PADDING, is_mask=True)
    predict_data = util.crete_patches_large(predict_data, constants.IMG_PATCH_SIZE, 16, constants.PADDING,is_mask=False)
    train_labels = util.one_hot_to_num(train_labels)

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
        steps=(constants.N_SAMPLES * constants.NUM_EPOCH) // constants.BATCH_SIZE)

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
