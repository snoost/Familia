import sys

from to_gensim import to_gensim_kv
from twe import TopicalWordEmbedding

twe = TopicalWordEmbedding(sys.argv[1])
twe.load_model()

kv = to_gensim_kv(twe.word_list, twe.word_embedding)

while True:
    word = input("Enter a word: ")
    print(kv.most_similar(positive=[word]))
