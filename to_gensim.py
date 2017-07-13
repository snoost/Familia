""" Gensim uses a special class called KeyedVectors to manage word vectors.
    This class comes with a lot of handy features. However, constructing it
    is a bit annoying because it has complicated internal representations.
    
    This class will serve as a reusable part for future word2vec experiments.


    class KeyedVectors(utils.SaveLoad):
        def __init__(self):
            self.syn0 = []
            self.syn0norm = None
            self.vocab = {}
            self.index2word = []
            self.vector_size = None

"""
from gensim.models.keyedvectors import Vocab, KeyedVectors
import numpy as np


def to_gensim_kv(word_list: list, word_embedding: dict):
    vocab_size = len(word_list)
    vector_size = len(word_embedding[word_list[0]])

    r = KeyedVectors()
    r.syn0 = np.zeros((vocab_size, vector_size), dtype=np.float32)
    r.vocab = {}
    r.index2word = list(word_list)

    for wid, word in enumerate(word_list):
        r.vocab[word] = Vocab(index=wid, count=None)
        r.syn0[wid] = np.asarray(word_embedding[word])
    return r



