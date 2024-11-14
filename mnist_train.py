import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
import mnist_inference
import os

# Configuration of neural network parameters
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "MNIST_model/"
MODEL_NAME = "mnist_model"


def train():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(-1, 28 * 28)
    x_test = x_test.reshape(-1, 28 * 28)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
    # Define input placeholders
    x = tf.compat.v1.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.compat.v1.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
    regularizer = tf.keras.regularizers.l2(REGULARIZATION_RATE)
    # Forward propagation to compute predictions
    y = mnist_inference.inference(x, regularizer)

    global_step = tf.Variable(0, trainable=False)
    # Define moving average operation
    variable_averages = tf.compat.v1.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.compat.v1.trainable_variables())
    # Compute cross-entropy loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # Compute total loss including regularization
    loss = cross_entropy_mean + tf.add_n(tf.compat.v1.get_collection('losses'))

    learning_rate = tf.compat.v1.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        x_train.shape[0] / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True
    )

    train_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
    # Initialize saver for saving the model
    saver = tf.compat.v1.train.Saver()

    # Start a TensorFlow session
    with tf.compat.v1.Session() as sess:
        tf.compat.v1.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            start = (i * BATCH_SIZE) % x_train.shape[0]
            end = min(start + BATCH_SIZE, x_train.shape[0])
            xs, ys = x_train[start:end], y_train[start:end]

            # Make sure ys is a NumPy array, not a Tensor
            if isinstance(ys, tf.Tensor):
                ys = ys.eval(session=sess)
            # Run the training operation and compute the loss
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            # Print loss every 1000 steps
            if i % 1000 == 0:
                print(f"After {step} training steps, loss on training batch is {loss_value:.4g}.")
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    # Ensure the model save path exists
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)
    train()


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.app.run()
