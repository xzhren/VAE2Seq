from tqdm import tqdm
import collections

vocab_file = "./corpus/reddit/vocab"
# VOCAB_SIZE = 200000

def make_vocab():
    with open("./corpus/reddit/train.txt") as f:
        pbar = tqdm(enumerate(f), total=3384185*2)
        vocab_counter = collections.Counter()
        for i, line in pbar: 
            if i % 2 ==0:
                info = line[len("post: "):]
            else:
                info = line[len("resp: "):]
            tokens = [_.strip() for _ in info.split(" ")]
            vocab_counter.update(tokens)

        print("Writing vocab file...")
        with open(vocab_file, 'w') as writer:
            for word, count in vocab_counter.most_common(len(vocab_counter)):
                writer.write(word + '\t' + str(count) + '\n')
        print("Finished writing vocab file")

import pandas as pd
def statistic_corpus():
    x_lens, y_lens = [], []
    with open("./corpus/reddit/train.txt") as f:
        pbar = tqdm(enumerate(f), total=3384185*2)
        for i, line in pbar: 
            if i % 2 ==0:
                info = line[len("post: "):]
                x_lens.append(len(info))
            else:
                info = line[len("resp: "):]
                y_lens.append(len(info))
    lens_pd = pd.DataFrame(x_lens, columns=["X_LENS"])
    lens_pd['Y_LENS'] = y_lens
    print(lens_pd.describe())


if __name__ == "__main__":
    # make_vocab()
    statistic_corpus()

"""
             X_LENS        Y_LENS
count  3.384185e+06  3.384185e+06
mean   9.807722e+01  9.690545e+01
std    4.627828e+01  4.681721e+01
min    5.000000e+00  4.000000e+00
25%    6.100000e+01  5.900000e+01
50%    9.400000e+01  9.200000e+01
75%    1.330000e+02  1.330000e+02
max    1.970000e+02  1.970000e+02

"""