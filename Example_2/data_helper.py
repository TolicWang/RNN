import numpy as np

start_token = 'B'
end_token = 'E'
# 由于VocabularyProcessor在处理的时候会自动去掉标点符号，所以在处理的时候要替换掉，等生成诗歌的时候在替换回来
D_token = 'D'  # 逗号
J_token = 'J'  # 句号
W_token = 'W'  # 问好
G_token = 'G'  # 感叹号


def process(fild_dir, max_length=70, sep=':'):
    """
    该函数的作用是把诗向量化
    :param fild_dir: 路径
    :param max_length: # 所能接受的古诗的最大长度（汉字+标点）
    :param sep: 分隔符，这儿为了同时兼容两个数据集 以poems.txt为训练集时spe=':'，以poetry.txt为训练集时sep=' '(空格)
    :return:

    example:
    输入必须每行为一首诗

    寒随穷律变，春逐鸟声开。
    初风飘带柳，晚雪间花梅。

    则对应为：
    [  1 235 297 ... 303 304 305]
    [  1 321 350 ... 470 263 471]

    """
    print("数据预处理中……")
    from tensorflow.contrib.learn.python.learn.preprocessing import VocabularyProcessor
    poems = []
    with open(fild_dir, encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = line.split(sep=sep)[-1]
            line = line.replace('，', D_token)
            line = line.replace('。', J_token)
            line = line.replace('？', W_token)
            content = line.replace('！', G_token)
            if len(content) > max_length or '（' in content:  # 所能接受的古诗的最大长度（汉字+标点）
                continue
            content = start_token + content + end_token
            poems.append(" ".join(content))
    # print(poems)
    vocab_processor = VocabularyProcessor(max_document_length=max_length,min_frequency=5)
    x = np.array(list(vocab_processor.fit_transform(poems)))
    dictionary = vocab_processor.vocabulary_.__dict__.copy()
    fre = dictionary['_freq']
    # print(sorted(fre.items(), key=lambda x: x[1], reverse=True))
    word_to_int = dictionary['_mapping']# {'<UNK>': 0, 'D': 1, 'J': 2, 'B': 3, 'E': 4, '不': 5, '人': 6}
    int_to_word = dictionary['_reverse_mapping']#['<UNK>', 'D', 'J', 'B', 'E', '不', '人',]er
    np.random.seed(50)
    shuffle_index = np.random.permutation(x.shape[0])
    shuffle_x = x[shuffle_index]
    shuffle_y = np.copy(shuffle_x)
    shuffle_y[:, :-1] = shuffle_x[:, 1:]
    # print(len(word_to_int))
    return shuffle_x, shuffle_y, word_to_int, int_to_word


def gen_batch(x, y, batch_size=256, index=0):
    start = index * batch_size
    end = start + batch_size
    return x[start:end], y[start:end]


def perfect_poem(poems):
    """
    本函数的作用是排版
    :param poems:
    :return:
    """

    p = ""
    for word in poems:
        p += word
        if word == J_token or word == G_token or word == W_token:
            p += '\n'
    p = p.replace(D_token, '，').replace(J_token, '。').replace(W_token, '？').replace(G_token, '！')
    return p


if __name__ == "__main__":
    # process('./data/poetry_test.txt')
    shuffle_x, shuffle_y, word_to_int, int_to_word=process('./data/poems.txt')
    # shuffle_x, shuffle_y, word_to_int, int_to_word = process('./data/poetry_test.txt', sep=' ', max_length=12)
    # print(word_to_int['，'])
    # print(shuffle_x.shape)
    print(word_to_int)
    print(int_to_word)
