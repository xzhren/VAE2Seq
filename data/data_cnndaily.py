import sklearn
import numpy as np
import tensorflow as tf
import random

from config import args

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

PAD_TOKEN = 0
UNK_TOKEN = 1
START_TOKEN = 2
END_TOKEN = 3
PAD_STRING = 'PAD'
UNK_STRING = 'UNK'
START_STRING = '<t>'
END_STRING = '</t>'

class CNNDAILY(BaseDataLoader):
    def __init__(self, batch_size, vocab_limit, max_input_len, max_output_len):
        super().__init__()
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.batch_size = batch_size
        self._load_vocab(vocab_limit)

    def _load_vocab(self, limit=0):
        word2idx = {"UNK":UNK_TOKEN, "PAD":PAD_TOKEN, "<S>":START_TOKEN, "</S>":END_TOKEN}
        idx2word = {UNK_TOKEN:"UNK", PAD_TOKEN:"PAD", START_TOKEN:"<S>", END_TOKEN:"</S>"}
        print(len(word2idx), len(idx2word))
        assert len(word2idx) == len(idx2word)
        counter = len(word2idx)
        with open("./corpus/cnndaily/vocab") as f:
            for i, line in enumerate(f):
                w, _ = line.split(" ")
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
    
    def load_data(self, fpath="./corpus/cnndaily/train"):
        batch_size = self.batch_size
        x_data, y_data = [], []
        atten_data = []
        src_data = []
        while True:
            with open(fpath+".txt.src") as fsrc, open(fpath+".txt.tgt.tagged") as ftgt:  
                for i, (src, tgt) in enumerate(zip(fsrc,ftgt)):
                    tokens_src = src.strip().split(" ")
                    tokens_ids = [self.word2idx[t] if t in self.word2idx else UNK_TOKEN for t in tokens_src]
                    x_data.append(tokens_ids)
                    tgt = tgt.split(START_STRING)[1]
                    tgt = tgt.split(END_STRING)[0].strip()
                    tokens_tgt = tgt.split(" ")
                    tokens_ids = [self.word2idx[t] if t in self.word2idx else UNK_TOKEN for t in tokens_tgt]
                    y_data.append(tokens_ids)

                    atten_label = np.zeros((self.max_output_len+1, self.max_input_len), dtype=int)
                    for i,t in enumerate(tokens_tgt[:self.max_output_len]):
                        for j,s in enumerate(tokens_src[:self.max_input_len]):
                            if s == t: atten_label[i][j] = 1
                    # print(np.sum(atten_label,1))
                    # print(tokens_tgt)
                    # print(tokens_src)
                    atten_data.append(atten_label)
                    src_data.append(tokens_src[:self.max_output_len])

        
                    if len(x_data) == batch_size:
                        assert len(x_data) == len(y_data)
                        assert len(x_data) == len(atten_data)
                        yield self._pad(x_data, y_data), atten_data, src_data
                        x_data, y_data = [], []
                        atten_data = []
                        src_data = []
                assert len(x_data) == len(y_data)
                assert len(x_data) == len(atten_data)
                if len(x_data) != 0:
                    yield self._pad(x_data, y_data), atten_data, src_data
                    x_data, y_data = [], []
            print("=====! EPOCH !======")
            break

    def _pad_one(self, data, maxlen):
        enc_inp = []
        dec_inp = []
        dec_out = []
        for x in data:
            if len(x) < maxlen:
                enc_inp.append(x + [PAD_TOKEN]*(maxlen-len(x)))
            else:
                enc_inp.append(x[:maxlen])

            y = x
            if len(y) < maxlen:
                dec_inp.append([START_TOKEN] + y + [PAD_TOKEN]*(maxlen-len(y)))
                dec_out.append(y + [END_TOKEN] + [PAD_TOKEN]*(maxlen-len(y)))
            else:
                dec_inp.append([START_TOKEN] + y[:maxlen])
                dec_out.append(y[:maxlen] + [END_TOKEN])
        return np.array(enc_inp), np.array(dec_inp), np.array(dec_out)

    def _pad(self, X, Y):
        x_enc_inp, x_dec_inp_full, x_dec_out = self._pad_one(X, self.max_input_len)
        y_enc_inp, y_dec_inp_full, y_dec_out = self._pad_one(Y, self.max_output_len)
        return x_enc_inp, x_dec_inp_full, x_dec_out, y_enc_inp, y_dec_inp_full, y_dec_out

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

    def trans_in_ref(self, finpath="./corpus/reddit/test.txt.tgt.tagged", foutpath="./saved/test.input.txt"):
        with open(finpath) as f, open(foutpath, "w") as fout:  
            for i, line in enumerate(f):
                info = line.split(END_STRING)[0]
                fout.write(info.strip()+"\n")

    def record_result(self, eval_log, finpath, frespaht, foutpath):
        with open(finpath+".txt.src") as fsrc, open(finpath+".txt.tgt.tagged") as ftgt, open(frespaht) as fres, open(foutpath, "w") as fout:  
            for i, (src, tgt) in enumerate(zip(fsrc,ftgt)):
                fout.write("sor: "+src.strip()+"\n")
                fout.write("ref: "+tgt.strip()+"\n")

                res = fres.readline()
                ress = ""
                for r in res.split(" "):
                    if r == END_STRING: break
                    ress += r + " "
                fout.write("res: "+ress.strip()+"\n")
                fout.write("-"*20+"\n")
            fout.write("\n\n\n")
            for metric, score in eval_log.items():
                if metric == "bleu":
                    fout.write("  bleu-1, bleu-2, bleu-3, bleu-4: %.5f,  %.5f,  %.5f,  %.5f\n" % score)
                elif metric == "rouge":
                    fout.write("  rouge-1, rouge-2, rouge-l: %.5f,  %.5f,  %.5f\n" % score)
                else:
                    fout.write("  %s: %.5f\n" % (metric, score))