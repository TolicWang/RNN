import tensorflow as tf
import os
from data_helper import gen_batch, process, start_token, end_token
import numpy as np

tf.app.flags.DEFINE_string('device_name', '/gpu:0', 'device num')
tf.app.flags.DEFINE_string('model_prefix', 'poems', 'model save prefix.')
tf.app.flags.DEFINE_string('model_path', './model', 'training model direction')
tf.app.flags.DEFINE_string('file_path', './data/poems.txt', 'training data direction')
tf.app.flags.DEFINE_integer('batch_size', 256, 'batch_size')
FLAGS = tf.app.flags.FLAGS


class CharRNNtoPoem:
    def __init__(self,
                 num_layer=2,
                 rnn_size=256,
                 batch_size=64,
                 embedding_size=128,
                 num_class=5000,
                 learning_rate=0.003,
                 epoches=500000,
                 every_epoch_to_save=5,
                 every_epoch_to_print=10,
                 rnn_model='lstm',
                 train=True):
        """
        :param num_layer:  模型网络层数
        :param rnn_size:    模型输出大小
        :param batch_size:
        :param embedding_size: 词向量维度
        :param num_class:       最终分类数
        :param learning_rate:
        :param epoches:
        :param every_epoch_to_save:
        :param every_epoch_to_print:
        :param model:
        :param train:
        """
        if train:
            self.batch_size = batch_size
        else:
            self.batch_size = 1
        self.num_layer = num_layer
        self.model = rnn_model
        self.rnn_size = rnn_size
        self.embedding_size = embedding_size
        self.num_class = num_class
        self.learning_rate = learning_rate
        self.epoches = epoches
        self.every_epoch_to_save = every_epoch_to_save
        self.every_epoch_to_print = every_epoch_to_print
        self.build_input()
        self.build_rnn()

    def build_input(self):
        """
        定义相关输入占位符
        :return:
        """
        with tf.name_scope('model_inputs'):
            self.inputs = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, None], name='input-x')
            self.targets = tf.placeholder(dtype=tf.int64, shape=[self.batch_size, None], name='input-y')
        with tf.name_scope('embedding_layer'):
            self.embedding = tf.Variable(tf.truncated_normal(shape=[self.num_class, self.embedding_size], stddev=0.1),
                                         name='embedding')
            self.model_inputs = tf.nn.embedding_lookup(self.embedding,
                                                       self.inputs)  # shape = [batch_size,time_step,embedding_size]

    def build_rnn(self):
        def get_a_cell(rnn_size, model='lstm'):
            if model == 'lstm':
                rnn_model = tf.nn.rnn_cell.BasicLSTMCell(num_units=rnn_size)
                """此处还可添加其他模型"""
            return rnn_model

        with tf.name_scope('build_rnn_model'):
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [get_a_cell(self.rnn_size) for _ in range(self.num_layer)])  # 搭建num_layer层的模型
            self.initial_state = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            self.outputs, self.final_state = tf.nn.dynamic_rnn(cell, inputs=self.model_inputs,
                                                               initial_state=self.initial_state)
            output = tf.reshape(self.outputs, [-1, self.rnn_size])
        with tf.name_scope('full_connection'):
            weights = tf.Variable(tf.truncated_normal(shape=[self.rnn_size, self.num_class]),
                                  name='weights')  # [128,5000]
            bias = tf.Variable(tf.zeros(shape=[self.num_class]), name='bias')
            self.logits = tf.nn.xw_plus_b(output, weights, bias, name='logits')

        with tf.name_scope('loss'):
            labels = tf.reshape(self.targets, [-1])
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=self.logits)
            self.loss = tf.reduce_mean(loss)
        with tf.name_scope('accuracy'):
            self.proba_prediction = tf.nn.softmax(self.logits, name='output_probability')
            self.prediction = tf.argmax(self.proba_prediction, axis=1, name='output_prediction')
            correct_predictions = tf.equal(self.prediction, labels)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def train(self, x, y):
        if not os.path.exists(FLAGS.model_path):
            os.makedirs(FLAGS.model_path)
        with tf.device(FLAGS.device_name):
            train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss=self.loss, name='optimaize')
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            print("模型初始化中……")
            sess.run(init_op)
            start_epoch = 0
            checkpoint = tf.train.latest_checkpoint(FLAGS.model_path)
            if checkpoint:
                saver.restore(sess, checkpoint)
                print("成功载入模型{0}".format(checkpoint))
                start_epoch += int(checkpoint.split('-')[-1])
            print("开始训练……")
            total_loss = 0
            try:
                for epoch in range(start_epoch, self.epoches):
                    n_chunk = len(x) // self.batch_size
                    # print(n_chunk)
                    ave_loss = total_loss / n_chunk
                    total_loss = 0
                    for batch in range(n_chunk):
                        x_batch, y_batch = gen_batch(x, y, self.batch_size, batch)
                        feed = {self.inputs: x_batch, self.targets: y_batch}
                        l, acc, _ = sess.run([self.loss, self.accuracy, train_op], feed_dict=feed)
                        total_loss += l

                        if batch % self.every_epoch_to_print == 0:
                            print('Epoch:%d, last epoch loss ave:%.5f  batch:%d, current epoch loss:%.5f, acc:%.3f' % (
                                epoch, ave_loss, batch, l, acc))
                    if epoch % self.every_epoch_to_save == 0:
                        print("保存模型……")
                        saver.save(sess, os.path.join(FLAGS.model_path, FLAGS.model_prefix),
                                   global_step=epoch)
            except KeyboardInterrupt:
                print("手动终止训练，保存模型……")
                saver.save(sess, os.path.join(FLAGS.model_path, FLAGS.model_prefix),
                           global_step=epoch)

    def compose_poem(self, begin_word, word_to_int, int_to_word, poem_max_len=24):
        """
        作诗
        :param begin_word:
        :param word_to_int:
        :param int_to_word:
        :param poem_max_len:
        :return:
        """
        x = np.array([list(map(word_to_int.get, start_token))])
        checkpoint = tf.train.latest_checkpoint(FLAGS.model_path)
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            if checkpoint:
                saver.restore(sess, checkpoint)
            else:
                print("请先训练好模型！")
                return
            preditcion_index, final_state = sess.run([self.prediction, self.final_state], feed_dict={self.inputs: x})
            if begin_word:
                word = begin_word
            else:
                word = int_to_word[preditcion_index[0]]
            poem_ = ''
            i = 0
            while word != end_token:
                poem_ += word
                i += 1
                if i >= poem_max_len:
                    break
                x[0, 0] = word_to_int[word]
                preditcion_index, final_state = sess.run([self.prediction, self.final_state],
                                                         feed_dict={self.inputs: x, self.initial_state: final_state})
                # print(preditcion_index[0])
                word = int_to_word[preditcion_index[0]]
        return poem_

#
if __name__ == "__main__":
    x, y, word_to_int, int_to_word = process(FLAGS.file_path)
    model = CharRNNtoPoem(batch_size=FLAGS.batch_size, num_class=len(word_to_int))
    model.train(x, y)
