import tensorflow as tf

# Number of input nodes (28x28 pixels)
INPUT_NODE = 784
# Number of output nodes (digits 0-9)
OUTPUT_NODE = 10
# Number of nodes in the hidden layer
LAYER1_NODE = 500

def get_weight_variable(shape, regularizer):
    weights = tf.compat.v1.get_variable(
        "weights", shape,
        initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1))
    if regularizer is not None:
        tf.compat.v1.add_to_collection('losses', regularizer(weights))
    return weights

def inference(input_tensor, regularizer):
    with tf.compat.v1.variable_scope('layer1', reuse=tf.compat.v1.AUTO_REUSE):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.compat.v1.get_variable(
            "biases", [LAYER1_NODE],
            initializer=tf.compat.v1.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.compat.v1.variable_scope('layer2', reuse=tf.compat.v1.AUTO_REUSE):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.compat.v1.get_variable(
            "biases", [OUTPUT_NODE],
            initializer=tf.compat.v1.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2
