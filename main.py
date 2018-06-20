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

    # Model Port

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 16, 16, 1]
    # Output Tensor Shape: [batch_size, 16, 16, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)


    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 16, 16, 32]
    # Output Tensor Shape: [batch_size, 8, 8, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 8, 8, 32]
    # Output Tensor Shape: [batch_size, 8, 8, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 8, 8, 64]
    # Output Tensor Shape: [batch_size, 4, 4, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 4, 4, 64]
    # Output Tensor Shape: [batch_size, 4 * 4 * 64]
    pool_shape = pool2.get_shape().as_list()
    pool2_flat = tf.reshape(pool2, [-1, pool_shape[1] * pool_shape[2] * pool_shape[3]])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 4 * 4 * 64]
    # Output Tensor Shape: [batch_size, 512]
    dense = tf.layers.dense(inputs=pool2_flat, units=512, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 512]
    # Output Tensor Shape: [batch_size, 2]
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
    train_data, train_labels = util.load_train_data(tiling=TILING)
    train_labels = util.one_hot_to_num(train_labels)

    # Create the Estimator
    road_estimator = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="outputs/road")

    # Train the model
    # TODO: Include data augmentation here via augmenting both train_data and train_labels
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
    eval_data, eval_labels = util.load_train_data(tiling=TILING)
    eval_labels = util.one_hot_to_num(eval_labels)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    eval_results = road_estimator.evaluate(input_fn=eval_input_fn)
    print(eval_results)

    # Do prediction on test data
    predict_data = util.load_test_data(tiling=TILING)

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
        img = util.label_to_img_inverse(608, 608, 16,  16, res[(i - 1) * offset:i * offset])
        img = util.img_float_to_uint8(img)
        Image.fromarray(img).save('predictions_test/' + file_names[i - 1])

    # Predictions Train
    predict_data, _ = util.load_train_data(tiling=True)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": predict_data},
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
    # TODO: Log some advanced metrics interesting for the report maybe?
    # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # run_metadata = tf.RunMetadata()
    # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    #     file_writer = tf.summary.FileWriter('<path>', sess.graph)
    #
    #     for (run_iteration...)
    #         ... = sess.run(....,
    #                        options=run_options,
    #                        run_metadata=run_metadata
    #                        )
    #
    #         file_writer.add_run_metadata(
    #             run_metadata, "run%d" % (run_iteration,), run_iteration)

    tf.app.run()
