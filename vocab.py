import logging
"""
# TODO; vocabulary abstraction, it includes
word to id mapping
id to word mapping
set of words
"""

logger = logging.getLogger("vocabulary")

def load_vocab(vocabf):
    vocabulary = {}
    with open(vocabf, 'r') as f:
        for line in f:
            items = line.split('\t')
            word = items[1]
            wid = int(items[2])
            vocabulary[wid] = word
    logger.info("Vocabulary load successfully! #vocab_size = %d" % len(vocabulary))
    return vocabulary
