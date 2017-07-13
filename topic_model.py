from collections import defaultdict
import logging

from vocab import load_vocab
logger = logging.getLogger('LDAModel')


class TopicModel:
    def __init__(self, vocabf, modelf):
        self.vocabf = vocabf
        self.modelf = modelf

        # Basic data structure
        self.vocabulary = load_vocab(self.vocabf)
        self.word_count = defaultdict(int)  # word to sum count
        self.topic_all_words_count = defaultdict(int)
        self.topic_to_words = {}

        self.load_model()

    def load_vocab(self):
        with open(self.vocabf, 'r') as f:
            for line in f:
                items = line.split('\t')
                word = items[1]
                wid = int(items[2])
                self.vocabulary[wid] = word
        logger.info("Vocabulary load successfully! #vocab_size = %d" % len(self.vocabulary))

    def load_model(self):
        """ essentially read the libsvm format """
        temp_topic_to_words = defaultdict(dict)

        with open(self.modelf, 'r') as f:
            for line in f:
                items = line.split(' ')
                wid = int(items[0])
                kvs = items[1:]
                for kv in kvs:
                    k, v = kv.split(":")
                    topic_id = int(k)
                    word_count = int(v)
                    self.topic_all_words_count[topic_id] += word_count
                    self.word_count[wid] += word_count
                    temp_topic_to_words[topic_id][wid] = word_count

        for topic, words in temp_topic_to_words.items():
            self.topic_to_words[topic] = sorted(words.items(), key=lambda x: x[1], reverse=True)

    def humanize(self, topic_id):
        data = self.topic_to_words[topic_id]
        return [(self.vocabulary[k[0]], k[1]/self.topic_all_words_count[topic_id]) for k in data[:10]]
