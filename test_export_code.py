from __future__ import print_function

import json
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import os

from data.data_code import CODE, TEST_DATA_SIZE, TRADIN_DATA_SIZE
from modules.vaeseq import VAESEQ
from config import args
from measures import evaluation_utils


def main():
    ## CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    ## Parameters
    if args.exp == "NONE":
        args.exp = args.graph_type
        
    exp_path = "./saved/"+args.exp+"/"
    args.training = False
    args.enc_max_len = 50
    args.dec_max_len = 30
    args.vocab_limit = 10000
    print(args)

    ## DataLoader
    dataloader = CODE(batch_size=args.batch_size, vocab_limit=args.vocab_limit, max_input_len=args.enc_max_len, max_output_len=args.dec_max_len)
    params = {
        'vocab_size': len(dataloader.idx2word),
        'word2idx': dataloader.word2idx,
        'idx2word': dataloader.idx2word,
        'token2idx': dataloader.token2idx,
        'idx2token': dataloader.idx2token,
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
        result_file = exp_path+"test_"+restore_path.split("-")[-1]+"_"

        # Test DataSet
        test_len = 20000
        batcher = dataloader.load_export_data(test_len)
        code_mean_np, code_logvar_np, desc_mean_np, desc_logvar_np = None, None, None, None
        for i in tqdm(range((test_len-1)//args.batch_size+1)):
            try:
                enc_inp, _, _, dec_inp, _, _ = next(batcher)
                # dec_inp = dataloader.update_word_dropout(dec_inp_full)
            except StopIteration:
                print("there are no more examples")
                break
            code_mean, code_logvar, desc_mean, desc_logvar = model.export_vectors(sess, enc_inp, dec_inp)
            if i == 0:
                code_mean_np, code_logvar_np, desc_mean_np, desc_logvar_np = code_mean, code_logvar, desc_mean, desc_logvar
            else:
                code_mean_np = np.vstack([code_mean_np, code_mean])
                desc_mean_np = np.vstack([desc_mean_np, desc_mean])
                code_logvar_np = np.vstack([code_logvar_np, code_logvar])
                desc_logvar_np = np.vstack([desc_logvar_np, desc_logvar])
        print("code_mean_np:", np.shape(code_mean_np), np.save(result_file+"code_mean.npy",code_mean_np))
        print("desc_mean_np:", np.shape(desc_mean_np), np.save(result_file+"desc_mean.npy",desc_mean_np))
        print("code_logvar_np:", np.shape(code_logvar_np), np.save(result_file+"code_logvar.npy",code_logvar_np))
        print("desc_logvar_np:", np.shape(desc_logvar_np), np.save(result_file+"desc_logvar.npy",desc_logvar_np))
    

if __name__ == '__main__':
    print(json.dumps(args.__dict__, indent=4))
    main()