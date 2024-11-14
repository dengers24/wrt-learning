import tensorflow as tf
from tensorflow.keras.datasets import mnist
import mnist_inference
import mnist_train
import time

# Disable eager execution
tf.compat.v1.disable_eager_execution()

EVAL_INTERVAL_SECS = 10

def evaluate():
    # Load the MNIST dataset
    (x_train, y_train), (x_validation, y_validation) = mnist.load_data()
    x_validation = x_validation / 255.0
    x_validation = x_validation.reshape(-1, 28 * 28)
    # Convert labels to one-hot encoding
    y_validation = tf.keras.utils.to_categorical(y_validation, num_classes=10)
    # Convert y_validation to a NumPy array to avoid TensorFlow tensor error
    y_validation = y_validation.astype('float32')
    # Define input placeholders
    x = tf.compat.v1.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.compat.v1.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
    # Compute the forward propagation result
    y = mnist_inference.inference(x, None)
    # Compute accuracy
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # Load the moving average variables
    variable_averages = tf.compat.v1.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.compat.v1.train.Saver(variables_to_restore)

    while True:
        with tf.compat.v1.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_score = sess.run(accuracy, feed_dict={x: x_validation, y_: y_validation})
                print(f"After {global_step} training steps, validation accuracy = {accuracy_score:.4g}")
            else:
                print('No checkpoint file found')
                return
        time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    evaluate()

if __name__ == '__main__':
    tf.compat.v1.app.run()
