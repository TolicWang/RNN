from CharRNNtoPoem import FLAGS, CharRNNtoPoem
from data_helper import process, perfect_poem

x, y, word_to_int, int_to_word = process(FLAGS.file_path)
model = CharRNNtoPoem(num_class=len(word_to_int), train=False)
print('请输入第一个字：')
word = input()
poem = model.compose_poem(begin_word=word, word_to_int=word_to_int, int_to_word=int_to_word, poem_max_len=24)
print(perfect_poem(poem))
