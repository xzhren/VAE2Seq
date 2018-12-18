from config import args

import sklearn
import numpy as np
import tensorflow as tf

import random

class BaseDataLoader(object):
    def __init__(self):
        self.enc_inp = None
        self.dec_inp = None # word dropout
        self.dec_out = None
        self.word2idx = None
        self.idx2word = None

    def next_batch(self):
        for i in range(0, len(self.enc_inp), args.batch_size):
            yield (self.enc_inp[i : i + args.batch_size],
                   self.dec_inp[i : i + args.batch_size],
                   self.dec_out[i : i + args.batch_size])

UNK_TOKEN = 0
PAD_TOKEN = 1
START_TOKEN = 2
END_TOKEN = 3
UNK_STRING = 'UNK'
PAD_STRING = 'PAD'
START_STRING = '<S>'
END_STRING = '</S>'

class REDDIT(BaseDataLoader):
    def __init__(self, batch_size, vocab_limit, max_input_len, max_output_len):
        super().__init__()
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self._load_vocab(vocab_limit)
        self.data_loader = self._load_data(batch_size)
        # self.enc_inp, self.dec_inp_full, self.dec_out = self._load_data(batch_size)
        # self.dec_inp = self._word_dropout(self.dec_inp_full)
        
    def next_batch(self):
        return next(self.data_loader)

    def _load_vocab(self, limit=0):
        word2idx = {"UNK":UNK_TOKEN, "PAD":PAD_TOKEN, "<S>":START_TOKEN, "</S>":END_TOKEN}
        idx2word = {UNK_TOKEN:"UNK", PAD_TOKEN:"PAD", START_TOKEN:"<S>", END_TOKEN:"</S>"}
        print(len(word2idx), len(idx2word))
        assert len(word2idx) == len(idx2word)
        counter = len(word2idx)
        with open("./corpus/reddit/vocab") as f:
            for i, line in enumerate(f):
                w, _ = line.split("\t")
                word2idx[w] = counter
                idx2word[counter] = w
                counter += 1
                if limit > 0 and i > limit: break
        print(len(word2idx), len(idx2word))
        assert len(word2idx) == len(idx2word)
        print("load vocab", len(word2idx), counter)
        print("sample vocab", list(word2idx.items())[:10])
        print("sample vocab", list(idx2word.items())[:10])

        self.word2idx = word2idx
        self.idx2word = idx2word
    
    def _load_data(self, batch_size):
        x_data, y_data = [], []
        while True:
            with open("./corpus/reddit/train.txt") as f:  
                for i, line in enumerate(f):
                    if i % 2 ==0:
                        info = line[len("post: "):]
                        tokens = [_.strip() for _ in info.split(" ")]
                        tokens_ids = [self.word2idx[t] if t in self.word2idx else UNK_TOKEN for t in tokens]
                        x_data.append(tokens_ids)
                    else:
                        info = line[len("resp: "):]
                        tokens = [_.strip() for _ in info.split(" ")]
                        tokens_ids = [self.word2idx[t] if t in self.word2idx else UNK_TOKEN for t in tokens]
                        y_data.append(tokens_ids)
                        assert len(x_data) == len(y_data)
                        if len(x_data) == batch_size:
                            yield self._pad(x_data, y_data)
                            x_data, y_data = [], []
            print("=====! EPOCH !======")
            break

    def _pad(self, X, Y):
        enc_inp = []
        dec_inp = []
        dec_out = []
        for x,y in zip(X, Y):
            if len(x) < self.max_input_len:
                enc_inp.append(x + [PAD_TOKEN]*(self.max_input_len-len(x)))
            else:
                enc_inp.append(x[:self.max_input_len])

            if len(y) < self.max_output_len:
                dec_inp.append([START_TOKEN] + y + [PAD_TOKEN]*(self.max_output_len-len(y)))
                dec_out.append(y + [END_TOKEN] + [PAD_TOKEN]*(self.max_output_len-len(y)))
            else:
                truncated = y[:self.max_output_len]
                dec_inp.append([START_TOKEN] + truncated)
                dec_out.append(truncated + [END_TOKEN])
        return np.array(enc_inp), np.array(dec_inp), np.array(dec_out)

    def _word_dropout(self, x):
        is_dropped = np.random.binomial(1, args.word_dropout_rate, x.shape)
        fn = np.vectorize(lambda x, k: UNK_TOKEN if (k and (x not in range(4))) else x)
        return fn(x, is_dropped)

    def shuffle(self):
        self.enc_inp, self.dec_inp, self.dec_out, self.dec_inp_full = sklearn.utils.shuffle(
            self.enc_inp, self.dec_inp, self.dec_out, self.dec_inp_full)

    def update_word_dropout(self, dec_inp_full):
        dec_inp = self._word_dropout(dec_inp_full)
        return dec_inp


def main():
    def word_dropout_test(d):
        print(' '.join(d.idx2word[idx] for idx in d.dec_inp_full[20]))
        print(' '.join(d.idx2word[idx] for idx in d.dec_inp[20]))

    def update_word_dropout_test(d):
        d.update_word_dropout()
        print(' '.join(d.idx2word[idx] for idx in d.dec_inp_full[20]))
        print(' '.join(d.idx2word[idx] for idx in d.dec_inp[20]))

    def next_batch_test(d):
        enc_inp, dec_inp, dec_out = next(d.next_batch())
        print(enc_inp.shape, dec_inp.shape, dec_out.shape)

    imdb = REDDIT()
    word_dropout_test(imdb)
    update_word_dropout_test(imdb)
    next_batch_test(imdb)


if __name__ == '__main__':
    main()
