import tensorflow as tf

# Define a TensorFlow session
sess = tf.Session()

# Verify the installation by printing a string
hello = tf.constant("Hello computer!")
print(sess.run(hello))

# Perform simple maths
a = tf.constant(20)
b = tf.constant(22)
print('a + b = {0}'.format(sess.run(a + b)))
