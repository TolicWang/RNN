# @Time    : 2019/4/4 11:08
# @Email  : wangchengo@126.com
# @File   : data_helper.py
# package version:
#               python 3.6
#               sklearn 0.20.0
#               numpy 1.15.2
#               tensorflow 1.5.0
import numpy as np
from tqdm import tqdm
import os
import h5py
import pickle

UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
PAD = "<pad>"
special_tokens = [UNK, SOS, EOS, PAD]
DATAPATH = os.path.dirname(os.path.abspath(__file__))[:-5]
CACHEPATH = os.path.join(DATAPATH, 'data', 'CACHE')


class input_data():
    def __init__(self,
                 special_tokens,
                 preview=True,
                 keep_rate=0.9,
                 share_vocab=False,
                 ):
        self.special_tokens = special_tokens
        self.preview = preview
        self.keep_rate = keep_rate
        self.share_vocab = share_vocab

    def _index_table_from_file(self, data):
        """
        构造字典
        :param data:
        :param special_tokens:
        :param split_token:
        :param preview:
        :return:
        """
        from collections import Counter
        all_words = [word for line in data for word in line]
        c = Counter()
        for item in tqdm(all_words, desc='### Making dictionary. ###'):
            if len(item) >= 1:
                c[item] += 1
        most_common_words = int(len(list(set(all_words))) * self.keep_rate)  ##

        all_uniq_words = []
        for (k, v) in tqdm(c.most_common(most_common_words), desc='### Making dictionary. ###'):
            all_uniq_words += [k]
        print('------------')
        # all_uniq_words.sort(key=all_words.index)
        all_uniq_words = special_tokens + all_uniq_words
        print('------------')
        vocab_table = {word: idx for idx, word in enumerate(all_uniq_words)}
        return vocab_table, vocab_table.__len__()

    def _create_vocab_tables(self, src_vocab_file, tgt_vocab_file):
        """
        为输入和输出分别创建字典/词表
        :param src_vocab_file:
        :param tgt_vocab_file:
        :param share_vocab:
        :param special_tokens:
        :param split_token:
        :param preview:
        :return:
        """
        src_vocab_table, src_vocab_table_len = self._index_table_from_file(src_vocab_file)
        if self.share_vocab:
            tgt_vocab_table = src_vocab_table
            tgt_vocab_table_len = src_vocab_table_len
        else:
            tgt_vocab_table, tgt_vocab_table_len = self._index_table_from_file(tgt_vocab_file)
        return src_vocab_table, tgt_vocab_table, src_vocab_table_len, tgt_vocab_table_len

    def _load_data(self, src_data_dir=None, tgt_data_dir=None):
        """
        载入原始数据
        :param src_data_dir:
        :param tgt_data_dir:
        :param preview:
        :return:
        """
        source_data = []
        with open(src_data_dir, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc='###  Loading source data.  ###'):
                line = line.strip('\n').split()
                source_data.append(line)
        target_data = []
        with open(tgt_data_dir, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc='###  Loading target data.  ###'):
                line = line.strip('\n').split()
                target_data.append(line)
        if self.preview:
            print('\n', '-' * 10, ' data preview', '-' * 10)
            print('source data:', source_data[:2])
            print('target data:', target_data[:2])
            print('-' * 10, ' data preview', '-' * 10, '\n\n\n')
        return source_data, target_data

    def _data_transform_to_index(self, data, vocab_table, marks=None):
        """
        将原始数据转换成索引
        :param data:
        :param vocab_table:
        :param split_token:
        :return:
        """
        all_index = []
        for line in tqdm(data, desc='### Indexing  ###'):
            temp = []
            for word in line:
                index = vocab_table.get(word)
                if index is not None:
                    temp.append(index)
                else:
                    temp.append(vocab_table[UNK])
            if marks is None:
                all_index.append(temp)
            elif marks == SOS:
                all_index.append([vocab_table[marks]] + temp)
            else:
                all_index.append(temp + [vocab_table[marks]])
        return all_index

    def _index_transform_to_data(self, index, tgt_vocab_table):
        vocabs = list(tgt_vocab_table.keys())
        translations = []
        for line in index:
            temp = []
            for word in line:
                temp.append(vocabs[word])
            translations.append(" ".join(temp))
        return translations

    def load_dataset(self, src_data_dir=None, tgt_data_dir=None):
        """
        载入处理好的数据
        :param src_data_dir:
        :param tgt_data_dir:
        :param preview:
        :param split_token:
        :param special_tokens:
        :return:
        """
        source_data, target_data = self._load_data(src_data_dir=src_data_dir, tgt_data_dir=tgt_data_dir)

        src_vocab_table, tgt_vocab_table, src_vocab_table_len, tgt_vocab_table_len \
            = self._create_vocab_tables(source_data, target_data)

        source_input = self._data_transform_to_index(source_data, src_vocab_table)
        target_input = self._data_transform_to_index(target_data, tgt_vocab_table, marks=SOS)
        target_output = self._data_transform_to_index(target_data, tgt_vocab_table, marks=EOS)
        return source_input, target_input, target_output, src_vocab_table, tgt_vocab_table, src_vocab_table_len, tgt_vocab_table_len

    def _padding(self, sentences, vocab_table):
        new_sens = []
        lengths = [len(item) for item in sentences]
        max_len = max(lengths)
        for sentence in sentences:
            s_len = len(sentence)
            if s_len < max_len:
                sentence += [vocab_table[PAD]] * (max_len - s_len)
            new_sens.append(sentence)
        return new_sens, lengths

    def gen_batch(self, src_in, tgt_in, tgt_out, src_vocab_table, tgt_vocab_table, batch_size=64):
        s_index, e_index, batches = 0, 0 + batch_size, len(src_in) // batch_size
        for i in range(batches):
            if e_index > len(src_in):
                e_index = len(src_in)
            batch_src_in = src_in[s_index: e_index]
            batch_tgt_in = tgt_in[s_index: e_index]
            batch_tgt_out = tgt_out[s_index: e_index]
            batch_src_in, source_lengths = self._padding(batch_src_in, src_vocab_table)
            batch_tgt_in, target_lengths = self._padding(batch_tgt_in, tgt_vocab_table)
            batch_tgt_out, _ = self._padding(batch_tgt_out, tgt_vocab_table)
            s_index, e_index = e_index, e_index + batch_size
            yield batch_src_in, batch_tgt_in, batch_tgt_out, source_lengths, target_lengths

    def cache(self, fname, var=None):
        var_name = list(var.keys())
        var['var_name'] = var_name
        file = open(fname, 'wb')
        pickle.dump(var, file)
        file.close()

    def read_cache(self, fname):
        file = open(fname, 'rb')
        data = pickle.load(file)
        file.close()
        var_name = data['var_name']
        temp = {}
        for name in var_name:
            temp[name] = data[name]
        return temp

    def load_data(self, src_data_dir, tgt_data_dir):
        fname = os.path.join(CACHEPATH, 'cache.pkl')
        if os.path.exists(fname):
            data = self.read_cache(fname)
            src_in = data['src_in']
            tgt_in = data['tgt_in']
            tgt_out = data['tgt_out']
            src_vocab_table = data['src_vocab_table']
            tgt_vocab_table = data['tgt_vocab_table']
            src_vocab_table_len = data['src_vocab_table_len']
            tgt_vocab_table_len = data['tgt_vocab_table_len']
            print("load '%s' successfully" % fname)
        else:
            if os.path.isdir(CACHEPATH) is False:
                os.makedirs(CACHEPATH)
            src_in, tgt_in, tgt_out, src_vocab_table, tgt_vocab_table, src_vocab_table_len, tgt_vocab_table_len \
                = self.load_dataset(src_data_dir, tgt_data_dir)
            self.cache(fname,
                       {'src_in': src_in, 'tgt_in': tgt_in, 'tgt_out': tgt_out, 'src_vocab_table': src_vocab_table,
                        'tgt_vocab_table': tgt_vocab_table, 'src_vocab_table_len': src_vocab_table_len,
                        'tgt_vocab_table_len': tgt_vocab_table_len})
        return src_in, tgt_in, tgt_out, src_vocab_table, tgt_vocab_table, src_vocab_table_len, tgt_vocab_table_len


if __name__ == '__main__':
    data = input_data(special_tokens=special_tokens, preview=True, keep_rate=0.9, share_vocab=False)
    source_input, target_input, target_output, src_vocab_table, tgt_vocab_table, src_vocab_table_len, tgt_vocab_table_len = \
        data.load_data(src_data_dir='../data/chinese.txt', tgt_data_dir='../data/english.txt', )
    batches = data.gen_batch(source_input, target_input, target_output, src_vocab_table, tgt_vocab_table,
                        32)
    for step, batch in enumerate(batches):
        print(step)
        pass
