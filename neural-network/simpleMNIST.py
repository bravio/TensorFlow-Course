#
# simpleMNIST.py
# Simple neural network to classify handwritten digits from the MNIST dataset.
#

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Use the TF helper function to pull down the data from the MNIST site
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Placeholder for 28x28px image data. The shape parameter None indicates that we know that
# this dimension exists, but we don't know how many items will be in this dimesion. 
# The 784 specifies each of these items will have 784 values. (28x28 = 784)
x = tf.placeholder(tf.float32, shape=[None, 784])

# y_ is called the "y bar" and is a 10 element vector, containing the predicted probability
# of each digit (0-9) class. Such as [0.14, 0.8, 0, 0, 0, 0, 0, 0, 0, 0.06]
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Define the weights and biases for each neuron that we will train
# Initialize the weights as a 784x10 tensor filled with 0s
W = tf.Variable(tf.zeros([784, 10]))
# We only have one bias per neuron, and that bias will have 10 values
b = tf.Variable(tf.zeros([10]))

# Define our model
# Softmax is our activation function. It's also exponential
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Loss function as cross entropy
# Softmax cross entropy with logits is the difference between our prediction
# and the actual values in the test data.
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# On each training step in gradient descent, we want to minimize cross entropy
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Initialize the global variables
init = tf.global_variables_initializer()

# Create an interactive session that can span multiple code blocks.
# We need to explicitly close the session with sess.close()
sess = tf.Session()

# Perform the initialization
sess.run(init)

# Perform 1000 training steps
for i in range(1000):
    # Get 100 random data points from the dataset. batch_xs = image. batch_ys = digit class
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # Test the model against the test data
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Evaluate how well the model did. Do this by comparing the digit with the highest probability
# in actual (y) and predicted (y_)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print("Test Accuracy: {0}%".format(test_accuracy * 100.0))

sess.close()
