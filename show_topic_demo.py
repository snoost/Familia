import sys
import pprint
import logging

from model import LDAModel

if __name__ == "__main__":
    logging.basicConfig()
    # print("Loading model ...")
    model = LDAModel('model/news/vocab_info.txt', 'model/news/news_lda.model')
    while True:
        topic_id = int(input("Topic Id (0-1999): "))
        pprint.pprint(model.humanize(topic_id))
