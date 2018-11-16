import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Define a path for TensorBoard log files
logPath = "./tb_logs/"

# Adds summary statistics to use in TensorBoard visualization
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

# Use the TF helper function to pull down the data from the MNIST site
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Using interactive session makes it the default session, so we don'tneed to pass sess as an argument
sess = tf.InteractiveSession()

# Define placeholders for the MNIST data
with tf.name_scope("MNIST_Input_Data"):
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y_")

# Change the MNISt input data from a list of values to a 28x28px x 1 grayscale value cube
# which the Convolutional NN can use
with tf.name_scope("MNIST_Input_Reshape"):
    x_image = tf.reshape(x, [-1, 28, 28, 1], name="x_image")
    tf.summary.image('input_img', x_image, 5)

# Define helper functions to create weight and bias variables, and convolution and pooling layers
# We are using RELU as our activation function. These must be initialized to a small positive number
# with some noise added, so we don't end up falling to 0 when comparing diffs
def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

# Convolution and Pooling
# We do convolution first, and then pooling. This is to control overfitting
def conv2d(x, W, name=None):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)

def max_pool_2x2(x, name=None):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

# Define the layers in the neural network

# 1st convolutional layer
# 32 features for each 5x5 patch of the image
with tf.name_scope('Conv_1'):
    with tf.name_scope('weights'):
        W_conv1 = weight_variable([5, 5, 1, 32], name="weight")
        variable_summaries(W_conv1)
    
    with tf.name_scope('biases'):
        b_conv1 = bias_variable([32], name="bias")
        variable_summaries(b_conv1)
    
    # Do convolution on images, add bias and push through RELU activation
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1, name="relu")
    tf.summary.histogram('h_conv1', h_conv1)
    # Take results and run through max_pool
    h_pool1 = max_pool_2x2(h_conv1, name="pool")

# 2nd convolutional layer
# Process the 32 features from Convolutional Layer 1, in a 5x5 patch. Return 64 features' weights and biases
with tf.name_scope('Conv_2'):
    with tf.name_scope('weights'):
        W_conv2 = weight_variable([5, 5, 32, 64], name="weight")
        variable_summaries(W_conv2)

    with tf.name_scope('biases'):
        b_conv2 = bias_variable([64], name="bias")
        variable_summaries(b_conv2)

    # Do convolution of the outpout from the 1st layer, and pool the results
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2, name="relu")
    tf.summary.histogram('h_conv2', h_conv2)
    # Take results and run through max_pool
    h_pool2 = max_pool_2x2(h_conv2, name="pool")

# Fully Connected Layer
with tf.name_scope('FC'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024], name="weight")
    b_fc1 = bias_variable([1024], name="bias")

    # Connect output of pooling layer 2 as input to the fully connected layer
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name="relu")

# Drop out some neurons to reduce overfitting
keep_prob = tf.placeholder(tf.float32, name="keep_prob")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer
with tf.name_scope('Readout'):
    W_fc2 = weight_variable([1024, 10], name="weight")
    b_fc2 = bias_variable([10], name="bias")

# Define the model
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Loss measurement
with tf.name_scope("Cross_Entropy"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

# Loss optimization
with tf.name_scope("Loss_Optimizer"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# What is correct
with tf.name_scope("Accuracy"):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    # How accurate is it?
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar("cross_entropy", cross_entropy)
tf.summary.scalar("training_accuracy", accuracy)

# TensorBoard - Merge Summaries
summarize_all = tf.summary.merge_all()

# Initialize all of the variables
sess.run(tf.global_variables_initializer())

# TensorBoard - Write the default grah out so we can view its structure
tbWriter = tf.summary.FileWriter(logPath, sess.graph)

# Train the model
num_steps = 2000
display_every = 100

# Start timer
start_time = time.time()
end_time = time.time()

for i in range(num_steps):
    batch = mnist.train.next_batch(50)
    _, summary = sess.run([train_step, summarize_all], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    # Periodic status display
    if i % display_every == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        end_time = time.time()
        print("step {0}, elapsed time {1:.2f} seconds, training accuracy {2:.3f}%".format(i, end_time - start_time, train_accuracy * 100.0))
        
        # Write the summaries to the log
        tbWriter.add_summary(summary, i)

# Display the summary
# Time to train
end_time = time.time()
print("Total training time for {0} batches: {1:.2f} seconds".format(i + 1, end_time - start_time))

# Accuracy on test data
print("Test accuracy: {0:.3f}%".format(
    accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}) * 100
))

sess.close()
