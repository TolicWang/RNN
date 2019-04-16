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

# [[1614, 4, 594, 122, 436, 4, 32697, 4, 54, 87, 1140, 1134, 278, 5390, 44, 6149, 967, 5, 695, 4, 430, 1767, 4, 3201, 32698, 4, 14171, 15368, 4, 83, 1552, 245, 48, 6423, 6149, 6], [9, 843, 5576, 4, 245, 9, 11060, 4, 18807, 4, 6046, 32699, 88, 32700, 4, 346, 21499, 4, 9036, 1474, 515, 1374, 608, 49, 4, 189, 515, 2761, 2520, 49, 6]]
# [[1614, 4, 594, 122, 436, 4, 32697, 4, 54, 87, 1140, 1134, 278, 5390, 44, 6149, 967, 5, 695, 4, 430, 1767, 4, 3201, 32698, 4, 14171, 15368, 4, 83, 1552, 245, 48, 6423, 6149, 6, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [9, 843, 5576, 4, 245, 9, 11060, 4, 18807, 4, 6046, 32699, 88, 32700, 4, 346, 21499, 4, 9036, 1474, 515, 1374, 608, 49, 4, 189, 515, 2761, 2520, 49, 6, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]]


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
                    inference=True)
    translation_index = model.infer(source_input,source_length)[0]
    translations = data._index_transform_to_data(translation_index, tgt_vocab_table)
    for sentence in translations:
        print(sentence[:-5])