import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os

OUTPUT_SIZE = 10
BATCH_SIZE = 32
TIME_STEP = 28
DIM = 28
LEARNING_RATE = 0.01
MODEL_SAVE_PATH = './data/model/'


def lstm(inputs):
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=OUTPUT_SIZE)
    h0 = cell.zero_state(batch_size=tf.shape(inputs)[1], dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs=inputs, initial_state=h0, time_major=True)
    return outputs[-1]


def train(mnist):
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='input-x')
    y = tf.placeholder(dtype=tf.int32, shape=[None], name='input-y')
    x_reshape = tf.reshape(x, shape=[-1, DIM, DIM], name='reshape-x')
    x_tranpose = tf.transpose(x_reshape, perm=[1, 0, 2], name='transpose-x')

    y_ = lstm(x_tranpose)
    with tf.name_scope('weighted-softmax'):
        weights = tf.Variable(tf.truncated_normal(shape=[OUTPUT_SIZE, OUTPUT_SIZE], stddev=0.1), dtype=tf.float32)
        bias = tf.Variable(tf.constant(0, shape=[OUTPUT_SIZE], dtype=tf.float32))
        logits = tf.nn.xw_plus_b(y_, weights, bias, name='softmax')
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(cross_entropy)
    global_step = tf.Variable(0, trainable=False)
    with tf.device('/gpu:0'):
        train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss, global_step=global_step)
    with tf.name_scope('accruacy'):
        predictions = tf.nn.in_top_k(predictions=logits, targets=y, k=1)
        accuracy = tf.reduce_mean(tf.cast(predictions, tf.float32))
    validation_feed = {x: mnist.validation.images, y: mnist.validation.labels}
    test_feed = {x: mnist.test.images, y: mnist.test.labels}
    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=5, )
        if os.path.exists(MODEL_SAVE_PATH + 'checkpoint'):
            saver.restore(sess, tf.train.latest_checkpoint(MODEL_SAVE_PATH))
            print('load model!\n')
        else:
            sess.run(tf.global_variables_initializer())
        for i in range(9999100000):
            xt, yt = mnist.train.next_batch(BATCH_SIZE)
            _, l, train_acc = sess.run([train_step, loss, accuracy], feed_dict={x: xt, y: yt})
            if i % 5000 == 0:
                val_acc = sess.run(accuracy, feed_dict=validation_feed)
                print("validatioin accuracy:%f" % val_acc)
            if i % 200 == 0:
                print("Loss on train:%f --- acc:%f" % (l, train_acc))
            if (i + 1) % 10000 == 0:
                saver.save(sess, MODEL_SAVE_PATH, global_step=global_step, write_meta_graph=False)


if __name__ == "__main__":

    mnist = input_data.read_data_sets('./data/MNIST_data')
    train(mnist)
