import tensorflow.contrib as contrib
import tensorflow as tf
from ConvLSTM import BasicConvLSTM
import numpy as np

# model = contrib.rnn.ConvLSTMCell
model = BasicConvLSTM

data = np.random.rand(64, 100, 10, 10, 28)
inputs = tf.placeholder(dtype=tf.float32, shape=[64, 100, 10, 10, 28])  # [batch_size,width,high,channeals] 5D
cell = model(conv_ndims=2, input_shape=[10, 10, 28], output_channels=2, kernel_shape=[3, 3])
initial_state = cell.zero_state(batch_size=100, dtype=tf.float32)
output, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, time_major=True, initial_state=initial_state)
print(output)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(output, feed_dict={inputs: data}))
