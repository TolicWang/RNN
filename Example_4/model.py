# @Time    : 2019/4/4 16:11
# @Email  : wangchengo@126.com
# @File   : model.py
# package version:
#               python 3.6
#               sklearn 0.20.0
#               numpy 1.15.2
#               tensorflow 1.5.0
import tensorflow as tf
import tensorflow.contrib as contrib
import os


def _create_or_load_embed(embed_name, vocab_size, embed_size, dtype):
    """
    建立词向量
    :param embed_name:
    :param vocab_size:
    :param embed_size:
    :param dtype:
    :return:
    """
    embedding = tf.get_variable(embed_name, [vocab_size, embed_size], dtype)
    return embedding


def create_emb_for_encoder_and_decoder(share_vocab,
                                       src_vocab_size,
                                       tgt_vocab_size,
                                       src_embed_size,
                                       tgt_embed_size,
                                       dtype=tf.float32,
                                       ):
    """
    为编码和解码的部分分别创建一个词向量
    :param share_vocab: 两者的词向量可以共享（一样）
    :param src_vocab_size: 单词维度
    :param tgt_vocab_size:
    :param src_embed_size: 词向量维度
    :param tgt_embed_size:
    :param dtype:
    :return:
    """
    if share_vocab:
        if src_vocab_size != tgt_vocab_size:
            raise ValueError("Share embedding but different src/tgt vocab sizes"
                             " %d vs. %d" % (src_vocab_size, tgt_vocab_size))
        assert src_embed_size == tgt_embed_size
        print("# Use the same embedding for source and target")
        embedding_encoder = _create_or_load_embed("embedding_share", vocab_size=src_vocab_size,
                                                  embed_size=src_embed_size, dtype=dtype)
        embedding_decoder = embedding_encoder
    else:
        with tf.variable_scope("encoder"):
            embedding_encoder = _create_or_load_embed("embedding_encoder", embed_size=src_embed_size,
                                                      vocab_size=src_vocab_size, dtype=dtype)
        with tf.variable_scope("decoder"):
            embedding_decoder = _create_or_load_embed("embedding_decoder", vocab_size=tgt_vocab_size,
                                                      embed_size=tgt_embed_size, dtype=dtype)
    return embedding_encoder, embedding_decoder


class Seq2Seq():
    def __init__(self,
                 encoder_rnn_size=32,
                 encoder_rnn_layer=2,
                 decoder_rnn_size=32,
                 decoder_rnn_layer=2,
                 batch_size=32,
                 epoches=100,
                 learning_rate=0.001,
                 share_vocab=False,
                 src_vocab_size=None,
                 tgt_vocab_size=None,
                 src_embed_size=64,
                 tgt_embed_size=64,
                 inference=False,
                 logger=None,
                 model_path=None
                 ):
        self.encoder_rnn_size = encoder_rnn_size  # 编码rnn的维度，即num_units
        self.encoder_rnn_layer = encoder_rnn_layer  # 编码rnn网络的堆叠层数
        self.decoder_rnn_size = decoder_rnn_size
        self.decoder_rnn_layer = decoder_rnn_layer
        self.batch_size = batch_size
        self.epoches = epoches
        self.learning_rate = learning_rate
        self.share_vocab = share_vocab
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.src_embed_size = src_embed_size
        self.tgt_embed_size = tgt_embed_size
        self.inference = inference
        self.logger = logger
        self.model_path = model_path

        assert self.encoder_rnn_size == self.decoder_rnn_size
        assert self.encoder_rnn_layer == self.decoder_rnn_layer
        self.logger.info('### Building Network...')
        self._build_placeholder()
        self._build_embedding()
        self._build_encoder()
        self._build_decoder()
        self._build_loss()

    def _build_placeholder(self):
        self.source_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name='src_in')  # 原始
        self.target_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name='tgt_in')
        self.target_output = tf.placeholder(dtype=tf.int32, shape=[None, None], name='tgt_out')
        self.encoder_inputs = tf.transpose(self.source_input)  # 因为后面设定了 time_major = True，所以此处要transpose
        self.decoder_inputs = tf.transpose(self.target_input)
        self.decoder_outputs = tf.transpose(self.target_output)

        self.source_lengths = tf.placeholder(dtype=tf.int32, shape=[None],
                                             name='source_seq_lengths')  # 原始输入中，每个样本（序列）的长度
        self.target_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name='target_seq_lengths')
        self.max_source_length = tf.reduce_max(self.source_lengths)
        self.max_target_length = tf.reduce_max(self.target_lengths)

    def _build_embedding(self):
        self.embedding_encoder, self.embedding_decoder = create_emb_for_encoder_and_decoder(
            share_vocab=self.share_vocab,
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            src_embed_size=self.src_embed_size,
            tgt_embed_size=self.tgt_embed_size)

    def _build_encoder(self):
        def get_encoder_cell(rnn_size):
            lstm_cell = tf.nn.rnn_cell.LSTMCell(rnn_size)
            return lstm_cell

        self.encoder_emb_inp = tf.nn.embedding_lookup(self.embedding_encoder, self.encoder_inputs)
        encoder_cell = tf.nn.rnn_cell.MultiRNNCell(
            [get_encoder_cell(self.encoder_rnn_size) for _ in range(self.encoder_rnn_layer)])
        self.encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(cell=encoder_cell,
                                                                           inputs=self.encoder_emb_inp,
                                                                           sequence_length=self.source_lengths,
                                                                           time_major=True,
                                                                           dtype=tf.float32)

    def _build_decoder(self):
        def get_decoder_cell(rnn_size):
            decoder_cell = tf.nn.rnn_cell.LSTMCell(rnn_size)  # 用来构造一个decoder_cell
            return decoder_cell

        decoder_cell = tf.nn.rnn_cell.MultiRNNCell(
            [get_decoder_cell(self.decoder_rnn_size) for _ in range(self.decoder_rnn_layer)])

        if self.inference:
            helper = contrib.seq2seq.GreedyEmbeddingHelper(self.embedding_decoder,
                                                           tf.fill([self.batch_size], 1), 2)
            maximum_iterations = tf.round(self.max_source_length * 2)# 这是谷歌NMT中提到的技巧，解码长度设为输入的2倍
        else:
            self.decoder_emb_inp = tf.nn.embedding_lookup(self.embedding_decoder, self.decoder_inputs)
            helper = contrib.seq2seq.TrainingHelper(inputs=self.decoder_emb_inp,
                                                    sequence_length=self.target_lengths,
                                                    time_major=True)
            maximum_iterations = self.max_target_length

        projection_layer = tf.layers.Dense(self.tgt_vocab_size, use_bias=False)  # 全连接层
        decoder = contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                               helper=helper,
                                               initial_state=self.encoder_final_state,
                                               output_layer=projection_layer)
        outputs, _, _ = contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                       output_time_major=True,
                                                       maximum_iterations=maximum_iterations)
        self.logits = outputs.rnn_output
        self.translations = tf.transpose(outputs.sample_id)

    def _build_loss(self):
        masks = tf.sequence_mask(self.target_lengths, self.max_target_length, dtype=tf.float32, name='masks')
        target_weights = tf.transpose(masks)  # padding 的部分不计入损失，所以要去掉
        # 因为设定了 time_major = True，所以此处要transpose
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.decoder_outputs, logits=self.logits)
        self.train_loss = (tf.reduce_sum(crossent * target_weights) / self.batch_size)

    def train(self, source_input, target_input, target_output, src_vocab_table, tgt_vocab_table, gen_batch):
        self.logger.info('### Training Network...')
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        params = tf.trainable_variables()

        gradients = tf.gradients(self.train_loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 2)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        update_step = optimizer.apply_gradients(zip(clipped_gradients, params))
        saver = tf.train.Saver(params, max_to_keep=10)
        model_name = 'train'

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            start_epoch = 0
            check_point = tf.train.latest_checkpoint(self.model_path)
            if check_point:
                saver.restore(sess, check_point)
                start_epoch += int(check_point.split('-')[-1])
                if start_epoch % 2 != 0:
                    start_epoch += 1
                self.logger.info("### Loading exist model <{}> successfully...".format(check_point))
            total_loss = 0
            try:
                for epoch in range(start_epoch, self.epoches):
                    n_chunk = len(source_input) // self.batch_size
                    ave_loss = total_loss / n_chunk
                    total_loss = 0
                    batches = gen_batch(source_input, target_input, target_output, src_vocab_table, tgt_vocab_table,
                                        self.batch_size)

                    for step, batch in enumerate(batches):
                        feed_dict = {self.source_input: batch[0],
                                     self.target_input: batch[1],
                                     self.target_output: batch[2],
                                     self.source_lengths: batch[3],
                                     self.target_lengths: batch[4]}
                        loss, _ = sess.run([self.train_loss, update_step], feed_dict=feed_dict)
                        total_loss += loss
                        if step % 5 == 0:
                            self.logger.info(
                                "### Epoch: [{}/{}]------batch: [{}/{}] -- Loss: {} -- last epoch ave loss: {} ".format(
                                    epoch, self.epoches, step, n_chunk,
                                    loss, ave_loss))
                    if epoch % 5 == 0:
                        self.logger.info("### Saving model {}...".format(model_name + '-' + str(epoch)))
                        saver.save(sess, os.path.join(self.model_path, model_name),
                                   global_step=epoch, write_meta_graph=False)
            except KeyboardInterrupt:
                self.logger.warning("KeyboardInterrupt saving...")
                saver.save(sess, os.path.join(self.model_path, model_name), global_step=epoch - 1,
                           write_meta_graph=False)

    def infer(self, inputs, source_length):
        self.logger.info('### Infering...')
        check_point = tf.train.latest_checkpoint(self.model_path)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            if not check_point:
                raise ValueError("Translation model doesn't exit, please training (with command python train.py) first")
            saver.restore(sess, check_point)
            feed_dict = {self.source_input: inputs,
                         self.source_lengths: source_length}
            translations = sess.run([self.translations], feed_dict=feed_dict)
            return translations


if __name__ == '__main__':
    model = Seq2Seq()
