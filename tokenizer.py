from vocab import load_vocab

max_word_len = 15


class SimpleTokenizer:
    """ Forward maximum matching (FMM) tokenizer"""

    def __init__(self, vocab):
        self.vocab = vocab

    @classmethod
    def is_english(char):
        return ord('A') <= ord(char) <= ord('Z') or \
               ord('a') <= ord(char) <= ord('z')

    def tokenize(self, doc):
        r = []
        i = 0
        text_len = len(doc)
        while i < len(doc):
            found_word = False
            char_list = []
            j = i
            while j < text_len and j < i + max_word_len:
                char_list.append(doc[j])
                word = ''.join(char_list)
                if word in self.vocab:
                    found_word = word
                j += 1

            if found_word:
                r.append(found_word)
                i += len(found_word)
            else:
                i += 1  # drop the char to tokenizer can advance.
        return r


if __name__ == "__main__":
    vocab = load_vocab('model/news/vocab_info.txt')
    word_set = set(vocab.values())
    t = SimpleTokenizer(word_set)
    doc = input("Input Doc: ")
    print(t.tokenize(doc))
