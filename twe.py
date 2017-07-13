"""
Topical Word Embedding (TWE) class. 
"""

import struct
import sys

from builtins import print
from typing import BinaryIO


def read_word(f: BinaryIO) -> str:
    r = b''
    while True:
        c = f.read(1)
        if c == b" ":
            break
        r += c
    return r.decode('utf-8').strip()


class TopicalWordEmbedding:
    def __init__(self, modelf):
        self.modelf = modelf

        self.word_list = []
        self.word_embedding = {}

        self.topic_list = []
        self.topic_embedding = {}

    def load_model(self):
        print("Loading Topical Word Embedding (TWE)...")
        FLOAT_SIZE = 4  # float32 == 4 bytes

        with open(self.modelf, 'rb') as f:
            l = f.readline()
            nb_word, nb_topic, dimension = map(int, l.split())
            print("#word = %d\t #topic = %d\t #emb_size = %d" % (nb_word, nb_topic, dimension))
            UNPACK_FORMAT = "128f"  # * dimension
            for _ in range(nb_word):
                word = read_word(f)
                emb = f.read(FLOAT_SIZE * dimension)
                vector = struct.unpack(UNPACK_FORMAT, emb)
                self.word_list.append(word)
                self.word_embedding[word] = vector

            for _ in range(nb_topic):
                l = f.readline()
                topic = read_word(f)
                emb = f.read(FLOAT_SIZE * dimension)
                vector = struct.unpack(UNPACK_FORMAT, emb)
                self.topic_list.append(topic)
                self.topic_embedding[topic] = vector

            print("Load Topical Word Embedding (TWE) successfully!")
