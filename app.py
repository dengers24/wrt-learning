import os
import glob
import tensorflow as tf
import mnist_inference
import mnist_train
from PIL import Image
import cv2

def image_prepare(image_path):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Use global threshold to remove noise
    ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # Convert OpenCV image to PIL image
    im = Image.fromarray(th1)
    # Convert to grayscale
    im = im.convert('L')
    # Resize the image to 28x28
    im = im.resize((28, 28), Image.LANCZOS)
    # Convert image to list
    im_list = list(im.getdata())
    # Invert image and normalize pixel values
    result = [(255 - x) * 1.0 / 255.0 for x in im_list]
    return result

def evaluate(image_array):
    x = tf.compat.v1.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y = mnist_inference.inference(x, None)
    prediction = tf.argmax(y, 1)
    variable_averages = tf.compat.v1.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.compat.v1.train.Saver(variables_to_restore)
    with tf.compat.v1.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            predicted_digit = sess.run(prediction, feed_dict={x: [image_array]})
            return predicted_digit[0]
        else:
            print('No checkpoint file found')
            return None

def main():
    # Directory containing the images to be processed
    image_dir = 'picture/'
    # Get a list of all image files in the directory
    image_paths = glob.glob(os.path.join(image_dir, '*'))
    for image_path in image_paths:
        # Prepare the image
        image_array = image_prepare(image_path)
        # Evaluate the image
        predicted_digit = evaluate(image_array)
        # Get the image filename
        image_name = os.path.basename(image_path)
        # Print the result
        print(f'{image_name} is predicted as: {predicted_digit}')

if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    main()
