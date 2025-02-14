import sklearn
import numpy as np
import tensorflow as tf
import random

import pickle
import tables
from tqdm import tqdm

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
END_TOKEN = 0
UNK_TOKEN = 1
START_TOKEN = 0
UNK_STRING = 'UNK'
# PAD_STRING = 'PAD'
START_STRING = '<S>'
END_STRING = '</S>'

TRADIN_DATA_SIZE = 2906408
TEST_DATA_SIZE = 9222

def get_hdfsf_item(offset, idx_h5, txt_h5):
    length, pos = idx_h5[offset]['length'], idx_h5[offset]['pos']
    txts = txt_h5[pos:pos + length].astype('int64')
    return txts

class CODE(BaseDataLoader):
    def __init__(self, batch_size, vocab_limit, max_input_len, max_output_len):
        super().__init__()
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.batch_size = batch_size
        self._load_vocab(vocab_limit)

    def _load_vocab(self, limit=0):
        data_dir = '/home/renxinzhang/renxingzhang/paper/deep-code-search/data/nodup/'
        vocab_tokens = pickle.load(open(data_dir+'vocab.tokens.pkl', 'rb'))
        vocab_desc = pickle.load(open(data_dir+'vocab.desc.pkl', 'rb'))
        vocab_tokens[START_STRING] = 0
        vocab_desc[START_STRING] = 0
        vocab_tokens[END_STRING] = 0
        vocab_desc[END_STRING] = 0
        vocab_tokens_dict = {}
        for k, v in vocab_tokens.items():
            vocab_tokens_dict[v] = k
        vocab_desc_dict = {}
        for k, v in vocab_desc.items():
            vocab_desc_dict[v] = k
        
        self.token2idx = vocab_tokens
        self.idx2token = vocab_tokens_dict
        self.word2idx = vocab_desc
        self.idx2word = vocab_desc_dict
        print("load vocab", len(self.word2idx), len(self.idx2word))
        print("load vocab", len(self.token2idx), len(self.idx2token))
        print("sample vocab", list(vocab_tokens.items())[:10])
        print("sample vocab", list(vocab_desc.items())[:10])
        

    def load_train_data(self, testflag=False):
        data_dir = '/home/renxinzhang/renxingzhang/paper/deep-code-search/data/nodup/'
        if testflag:
            table_tokens = tables.open_file(data_dir+'test.tokens.h5')
            table_desc = tables.open_file(data_dir+'test.desc.h5')
            DATA_SIZE = TEST_DATA_SIZE
        else:
            table_tokens = tables.open_file(data_dir+'train.tokens.h5')
            table_desc = tables.open_file(data_dir+'train.desc.h5')
            DATA_SIZE = TRADIN_DATA_SIZE
        print(table_tokens)
        print(table_desc)
        txt_tokens = table_tokens.get_node('/phrases')
        idx_tokens = table_tokens.get_node('/indices')
        txt_desc = table_desc.get_node('/phrases')
        idx_desc = table_desc.get_node('/indices')

        batch_size = self.batch_size
        x_data, y_data = [], []
        while True:
            for i in range(DATA_SIZE):
                tokens = get_hdfsf_item(i, idx_tokens, txt_tokens)
                desc = get_hdfsf_item(i, idx_desc, txt_desc)
                x_data.append(tokens)
                y_data.append(desc)
                if len(x_data) == batch_size:
                    assert len(x_data) == len(y_data)
                    yield self._pad(x_data, y_data)
                    x_data, y_data = [], []
            assert len(x_data) == len(y_data)
            if len(x_data) != 0:
                yield self._pad(x_data, y_data)
                x_data, y_data = [], []
            print("=====! EPOCH !======")
            break

    def load_export_data(self, DATA_SIZE):
        data_dir = './corpus/code/'
        table_tokens = tables.open_file(data_dir+'test2.tokens.h5')
        table_desc = tables.open_file(data_dir+'test2.desc.h5')
        print(table_tokens)
        print(table_desc)
        txt_tokens = table_tokens.get_node('/phrases')
        idx_tokens = table_tokens.get_node('/indices')
        txt_desc = table_desc.get_node('/phrases')
        idx_desc = table_desc.get_node('/indices')

        batch_size = self.batch_size
        x_data, y_data = [], []
        
        for i in range(DATA_SIZE):
            tokens = get_hdfsf_item(i, idx_tokens, txt_tokens)
            desc = get_hdfsf_item(i, idx_desc, txt_desc)
            x_data.append(tokens)
            y_data.append(desc)
            if len(x_data) == batch_size:
                assert len(x_data) == len(y_data)
                yield self._pad(x_data, y_data)
                x_data, y_data = [], []
        assert len(x_data) == len(y_data)
        if len(x_data) != 0:
            yield self._pad(x_data, y_data)
            x_data, y_data = [], []
        print("=====! EPOCH !======")

    def _pad_one(self, data, maxlen):
        enc_inp = []
        dec_inp = []
        dec_out = []
        for x in data:
            y = x
            if len(x) < maxlen:
                x = np.append(x, [PAD_TOKEN]*maxlen)
                yin = np.append([START_TOKEN], y)
                yin = np.append(yin, [PAD_TOKEN]*maxlen)
                yout = np.append(y, [END_TOKEN])
                yout = np.append(yout, [PAD_TOKEN]*maxlen)
            else:
                yin = np.append([START_TOKEN], y)
                yout = np.append(y[:maxlen], [END_TOKEN])

            enc_inp.append(x[:maxlen])
            dec_inp.append(yin[:maxlen+1])
            dec_out.append(yout[:maxlen+1])
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

    def trans_in_ref(self, foutpath="./saved/test.input.txt"):
        data_dir = '/home/renxinzhang/renxingzhang/paper/deep-code-search/data/nodup/'
        table_tokens = tables.open_file(data_dir+'test.tokens.h5')
        table_desc = tables.open_file(data_dir+'test.desc.h5')
        print(table_tokens)
        print(table_desc)
        txt_tokens = table_tokens.get_node('/phrases')
        idx_tokens = table_tokens.get_node('/indices')
        txt_desc = table_desc.get_node('/phrases')
        idx_desc = table_desc.get_node('/indices')

        with open(foutpath, "w") as fout:  
            for i in range(TEST_DATA_SIZE):
                # tokens = get_hdfsf_item(i, idx_tokens, txt_tokens)
                desc = get_hdfsf_item(i, idx_desc, txt_desc)
                desc = [self.idx2word[d] for d in desc]    
                fout.write(" ".join(desc)+"\n")

    def record_result(self, eval_log, frespaht, foutpath):
        data_dir = '/home/renxinzhang/renxingzhang/paper/deep-code-search/data/nodup/'
        table_tokens = tables.open_file(data_dir+'test.tokens.h5')
        table_desc = tables.open_file(data_dir+'test.desc.h5')
        print(table_tokens)
        print(table_desc)
        txt_tokens = table_tokens.get_node('/phrases')
        idx_tokens = table_tokens.get_node('/indices')
        txt_desc = table_desc.get_node('/phrases')
        idx_desc = table_desc.get_node('/indices')

        with open(frespaht) as fres, open(foutpath, "w") as fout:  
            for i in range(TEST_DATA_SIZE):
                tokens = get_hdfsf_item(i, idx_tokens, txt_tokens)
                desc = get_hdfsf_item(i, idx_desc, txt_desc)
                tokens = [self.idx2token[t] for t in tokens]  
                desc = [self.idx2word[d] for d in desc]    
                fout.write("sor: "+" ".join(tokens)+"\n")
                fout.write("ref: "+" ".join(desc)+"\n")

                res = fres.readline()
                ress = ""
                for r in res.split(" "):
                    if r == "</S>": break
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