# @Time    : 2019/4/12 10:20
# @Email  : wangchengo@126.com
# @File   : inference.py
# package version:
#               python 3.6
#               sklearn 0.20.0
#               numpy 1.15.2
#               tensorflow 1.5.0

from utils.data_helper import input_data
from utils.logs import Logger
import logging
from model import Seq2Seq
from utils.data_helper import special_tokens

if __name__ == '__main__':
    data = input_data(special_tokens=special_tokens, preview=True, keep_rate=1.0, share_vocab=False)
    _, _, _, src_vocab_table, tgt_vocab_table, src_vocab_table_len, tgt_vocab_table_len = \
        data.load_data(src_data_dir='./data/chinese.txt', tgt_data_dir='./data/english.txt')
    logger = Logger(log_file_name='./log_inference.txt', log_level=logging.DEBUG, logger_name="test").get_log()
    # print("Please type a Chinese sentence with one space in each phrase: \n")
    # while True:
    sentences = [["我 来自 中国 , 我 是 中国人 ."],
                 ["我国 自行 研制 了 功能 强大 的 机群 操作系统 , 配 有 多种 流行 的 高级 程序 语言 , 主流 并行 编程 环境 和 工具 ."]]
    s = [item[0].split() for item in sentences]
    index = data._data_transform_to_index(s, src_vocab_table)
    source_input, source_length = data._padding(index, src_vocab_table)
    model = Seq2Seq(src_vocab_size=src_vocab_table_len,
                    tgt_vocab_size=tgt_vocab_table_len,
                    logger=logger,
                    batch_size=len(source_input),
                    model_path='./MODEL',
                    inference=True,
                    use_attention=True)
    translation_index = model.infer(source_input,source_length)[0]
    translations = data._index_transform_to_data(translation_index, tgt_vocab_table)
    for sentence in translations:
        print(sentence[:-5])