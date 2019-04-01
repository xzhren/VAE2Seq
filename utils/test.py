# import tensorflow as tf
# with tf.variable_scope("scope1") as scope1:
#     with tf.variable_scope("/scope2") as scope2:
#         print(scope2.name)

# import numpy as np

# inti_a_na = np.random.rand(10,5)
# print(inti_a_na, inti_a_na.shape)
# a = tf.convert_to_tensor(inti_a_na, dtype=tf.float32)
# b = a + 1
# c = b + 1
# with tf.Session() as sess:
#     print(c.eval())
#     inti_b_na = np.random.rand(10,5)
#     print(inti_b_na)
#     res = sess.run(c, {b:inti_b_na})
#     print(res)

import tables
import pickle

def get_hdfsf_item(offset, idx_h5, txt_h5):
    length, pos = idx_h5[offset]['length'], idx_h5[offset]['pos']
    txts = txt_h5[pos:pos + length].astype('int64')
    return txts


data_dir = '/home/renxinzhang/renxingzhang/paper/deep-code-search/data/nodup/'
vocab_tokens = pickle.load(open(data_dir+'vocab.tokens.pkl', 'rb'))
vocab_desc = pickle.load(open(data_dir+'vocab.desc.pkl', 'rb'))
# vocab_tokens[START_STRING] = 0
# vocab_desc[START_STRING] = 0
# vocab_tokens[END_STRING] = 0
# vocab_desc[END_STRING] = 0
vocab_tokens_dict = {}
for k, v in vocab_tokens.items():
    vocab_tokens_dict[v] = k
vocab_desc_dict = {}
for k, v in vocab_desc.items():
    vocab_desc_dict[v] = k

# print(vocab_tokens)
# print("s", vocab_tokens['s'])

# self.token2idx = vocab_tokens
# self.idx2token = vocab_tokens_dict
# self.word2idx = vocab_desc
# self.idx2word = vocab_desc_dict

data_dir = '/home/renxinzhang/renxingzhang/paper/deep-code-search/data/nodup/'

table_tokens = tables.open_file(data_dir+'test.tokens.h5')
table_desc = tables.open_file(data_dir+'test.desc.h5')
# print(table_tokens)
# print(table_desc)
txt_tokens = table_tokens.get_node('/phrases')
idx_tokens = table_tokens.get_node('/indices')
txt_desc = table_desc.get_node('/phrases')
idx_desc = table_desc.get_node('/indices')

with open(data_dir+"train_raw_code.txt") as rawcodefile:
    for i in range(9000):
        tokens = get_hdfsf_item(i, idx_tokens, txt_tokens)
        desc = get_hdfsf_item(i, idx_desc, txt_desc)
        # print(i)
        # print(" ".join([vocab_tokens_dict[t] for t in tokens]))
        print(" ".join([vocab_desc_dict[t] for t in desc]))
        # print(rawcodefile.readline())