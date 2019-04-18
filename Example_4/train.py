# @Time    : 2019/4/9 19:09
# @Email  : wangchengo@126.com
# @File   : train.py
# package version:
#               python 3.6
#               sklearn 0.20.0
#               numpy 1.15.2
#               tensorflow 1.5.0
from utils.data_helper import special_tokens
from utils.data_helper import input_data
from utils.logs import Logger
import logging
from model import Seq2Seq

if __name__ == '__main__':
    logger = Logger(log_file_name='./log_train.txt', log_level=logging.DEBUG, logger_name="test").get_log()
    data = input_data(special_tokens=special_tokens, preview=True, keep_rate=1.0, share_vocab=False)
    source_input, target_input, target_output, src_vocab_table, tgt_vocab_table, src_vocab_table_len, tgt_vocab_table_len = \
        data.load_data(src_data_dir='./data/chinese.txt', tgt_data_dir='./data/english.txt')
    model = Seq2Seq(src_vocab_size=src_vocab_table_len,
                    tgt_vocab_size=tgt_vocab_table_len,
                    logger=logger,
                    model_path='./MODEL',
                    batch_size=32,
                    use_attention=True)
    model.train(source_input, target_input, target_output, src_vocab_table, tgt_vocab_table, data.gen_batch)
