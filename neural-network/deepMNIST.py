import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Use the TF helper function to pull down the data from the MNIST site
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Using interactive session makes it the default session, so we don'tneed to pass sess as an argument
sess = tf.InteractiveSession()

# Define placeholders for the MNIST data
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Change the MNISt input data from a list of values to a 28x28px x 1 grayscale value cube
# which the Convolutional NN can use
x_image = tf.reshape(x, [-1, 28, 28, 1], name="x_image")

# Define helper functions to create weight and bias variables, and convolution and pooling layers
# We are using RELU as our activation function. These must be initialized to a small positive number
# with some noise added, so we don't end up falling to 0 when comparing diffs
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Convolution and Pooling
# We do convolution first, and then pooling. This is to control overfitting
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Define the layers in the neural network

# 1st convolutional layer
# 32 features for each 5x5 patch of the image
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
# Do convolution on images, add bias and push through RELU activation
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# Take results and run through max_pool
h_pool1 = max_pool_2x2(h_conv1)

# 2nd convolutional layer
# Process the 32 features from Convolutional Layer 1, in a 5x5 patch. Return 64 features' weights and biases
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
# Do convolution of the outpout from the 1st layer, and pool the results
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Fully Connected Layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

# Connect output of pooling layer 2 as input to the fully connected layer
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Drop out some neurons to reduce overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# Define the model
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Loss measurement
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

# Loss optimization
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# What is correct
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# How accurate is it?
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize all of the variables
sess.run(tf.global_variables_initializer())

# Train the model
num_steps = 3000
display_every = 100

# Start timer
start_time = time.time()
end_time = time.time()

for i in range(num_steps):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    # Periodic status display
    if i % display_every == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        end_time = time.time()
        print("step {0}, elapsed time {1:.2f} seconds, training accuracy {2:.3f}%".format(i, end_time - start_time, train_accuracy * 100.0))

# Display the summary
# Time to train
end_time = time.time()
print("Total training time for {0} batches: {1:.2f} seconds".format(i + 1, end_time - start_time))

# Accuracy on test data
print("Test accuracy: {0:.3f}%".format(
    accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}) * 100
))

sess.close()
