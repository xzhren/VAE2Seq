from __future__ import print_function

import json
import tensorflow as tf
from tqdm import tqdm

from data.data_reddit import REDDIT
from modules.vaeseq import VAESEQ
from config import args
from measures import evaluation_utils

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    ## Parameters
    args.max_len = 20
    args.batch_size = 64
    args.max_dec_len = args.max_len+1
    # args.display_info_step = 10000
    args.rnn_size = 256
    args.latent_size = 256
    print(args)
    exp_path = "./saved/vaeseq_trans/"

    ## DataLoader
    dataloader = REDDIT(batch_size=args.batch_size, vocab_limit=35000, max_input_len=args.max_len, max_output_len=args.max_len)
    params = {
        'vocab_size': len(dataloader.word2idx),
        'word2idx': dataloader.word2idx,
        'idx2word': dataloader.idx2word,}
    print('Vocab Size:', params['vocab_size'])

    ## ModelInit    
    model = VAESEQ(params)

    ## Session
    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    restore_path = tf.train.latest_checkpoint(exp_path)
    saver.restore(sess, restore_path)
    print("Model restore from file: %s" % (restore_path))
        
    # Parpear Dir
    ref_file = exp_path+"vae.test.input.txt"
    trans_file = exp_path+"vae.test.output.txt"
    result_file = exp_path+"vae.test.result.txt"
    test_file = "./corpus/reddit/test.txt"

    # Test Dir
    dataloader.trans_in_ref(finpath=test_file, foutpath=ref_file)
    with open(trans_file, "w") as f:
        f.write("")
    print("[PAEPEAR DATASET]")

    # Test DataSet
    test_len = 20000
    batcher = dataloader.load_data(fpath=test_file)
    for _ in tqdm(range((test_len-1)//args.batch_size+1)):
        try:
            enc_inp, _, _, _, _, _ = next(batcher)
            # dec_inp = dataloader.update_word_dropout(dec_inp_full)
        except StopIteration:
            print("there are no more examples")
            break
        # model.evaluation(sess, enc_inp, trans_file)
        model.evaluation_encoder_vae(sess, enc_inp, trans_file)

    # Evaluation
    eval_log = {}
    for metric in ['bleu','rouge','accuracy','word_accuracy']:
        score = evaluation_utils.evaluate(
            ref_file,
            trans_file,
            metric)
        eval_log[metric] = score
        print("  %s: %.1f" % (metric, score))

    # Record Log
    dataloader.record_result(eval_log, finpath=test_file, frespaht=trans_file, foutpath=result_file)
    

if __name__ == '__main__':
    print(json.dumps(args.__dict__, indent=4))
    main()