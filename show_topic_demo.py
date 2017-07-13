import pprint
import logging

from topic_model import TopicModel

if __name__ == "__main__":
    logging.basicConfig()
    # print("Loading model ...")
    model = TopicModel(vocabf='model/news/vocab_info.txt',
                       modelf='model/news/news_lda.model')
    while True:
        topic_id = int(input("Topic Id (0-1999): "))
        pprint.pprint(model.humanize(topic_id))
