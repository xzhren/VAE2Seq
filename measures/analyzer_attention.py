from __future__ import print_function

import json
import tensorflow as tf
from tqdm import tqdm
import os

import sys
sys.path.append("./")

from data.data_iwslt import IWSLT
from modules.vaeseq import VAESEQ
from config import args
from measures import evaluation_utils

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.switch_backend('agg')
plt.rcParams['font.family']=['SimHei'] 

def save_heatmap_fig(xlabels, ylabels, values, filename):
    fig, ax = plt.subplots()
#     a = np.random.uniform(0, 1, size=(10, 10))
    sns.heatmap(values, cmap='Blues', ax=ax)
    # sns.heatmap(a, cmap='Blues', linewidth=0.5)
    
    ax.set_xticklabels(xlabels)
    # for tick in ax.get_xticklabels():
    #     tick.set_rotation(90)
    # plt.rcParams['font.family']=['DejaVu Sans'] 
    ax.set_yticklabels(ylabels)
    for tick in ax.get_yticklabels():
        tick.set_rotation(0)
    # plt.xticks(rotation=90)
    # plt.yticks(rotation=90)
    plt.savefig("logs/"+filename, bbox_inches='tight')
#     plt.show()

def main():
    ## CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    ## Parameters
    if args.exp == "NONE":
        args.exp = args.graph_type
    args.enc_max_len =  100
    args.dec_max_len = 100
    args.vocab_limit = 35000
    args.batch_size = 10
    exp_path = "./saved/"+args.exp+"/"
    args.training = False
    test_len = 1261
    args.data_len = test_len
    args.diff_input = True
    print(args)

    ## DataLoader
    dataloader = IWSLT(batch_size=args.batch_size, vocab_limit=args.vocab_limit, max_input_len=args.enc_max_len, max_output_len=args.dec_max_len)
    params = {
        'vocab_size_encoder': len(dataloader.idx2token),
        'vocab_size': len(dataloader.word2idx),
        'word2idx': dataloader.word2idx,
        'idx2word': dataloader.idx2word,
        'idx2token': dataloader.idx2token,
        'token2id': dataloader.token2idx,
        'loss_type': args.loss_type,
        'graph_type': args.graph_type}
    print('Vocab Size:', params['vocab_size'])

    ## ModelInit    
    model = VAESEQ(params)

    ## Session
    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        restore_path = tf.train.latest_checkpoint(exp_path)
        saver.restore(sess, restore_path)
        print("Model restore from file: %s" % (restore_path))
        
        # Parpear Dir
        ref_file = exp_path+"test.input.txt"
        trans_file = exp_path+"test.output.txt"
        result_file = exp_path+"test."+restore_path.split("-")[-1]+".result.txt"
        test_file = "./corpus/iwslt2015/prepro/test.en"

        # Test Dir
        dataloader.trans_in_ref(finpath=test_file, foutpath=ref_file)
        with open(trans_file, "w") as f:
            f.write("")
        print("[PAEPEAR DATASET]")

        # Test DataSet
        test_file = "./corpus/iwslt2015/prepro/test"
        batcher = dataloader.load_data(fpath=test_file)
        for _ in tqdm(range((test_len-1)//args.batch_size+1)):
            try:
                (enc_inp, _, _, dec_inp, _, _), x_enc_inp_oovs, data_oovs, _ = next(batcher)
                # enc_inp, _, _, _, _, _ = next(batcher)
                # dec_inp = dataloader.update_word_dropout(dec_inp_full)
                max_oovs_len = 0 if len(data_oovs) == 0 else max([len(oov) for oov in data_oovs]) 
            except StopIteration:
                print("there are no more examples")
                break
            # model.evaluation(sess, enc_inp, trans_file, x_enc_inp_oovs, max_oovs_len, data_oovs)
            result = model.export_attentions(sess, enc_inp, dec_inp)

            for i, (att, xtxt, ytxt) in enumerate(result):
                xtxt = [i for i in xtxt if i != 'PAD']
                ytxt = [i for i in ytxt if i != '</S>']
                print(att.shape, len(xtxt), len(ytxt))
                att = att[:len(ytxt), :len(xtxt)]
                print(xtxt, ytxt)
                print(att.shape, len(xtxt), len(ytxt))
                save_heatmap_fig(list(xtxt), list(ytxt), att, "heatmap_"+str(i)+".pdf")

            break


if __name__ == '__main__':
    print(json.dumps(args.__dict__, indent=4))
    main()