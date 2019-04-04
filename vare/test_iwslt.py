from __future__ import print_function

import json
import tensorflow as tf
from tqdm import tqdm
import os

# from data_reddit import REDDIT
from model_vae import VRAE
from config import args
from measures import evaluation_utils

import sys
sys.path.append(".")
from data.data_iwslt import IWSLT

def main():
    ## CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    args.max_len = 100
    args.batch_size = 64
    args.max_dec_len = 100
    args.display_info_step = 10000
    args.isPointer = False
    args.vocab_limit = 35000
    test_len = 1261
    print(args)

    ## Parameters
    dataloader = IWSLT(batch_size=args.batch_size, vocab_limit=args.vocab_limit, max_input_len=args.max_len, max_output_len=args.max_dec_len)
    params = {
        'vocab_size': len(dataloader.word2idx),
        'word2idx': dataloader.word2idx,
        'idx2word': dataloader.idx2word,}
    print('Vocab Size:', params['vocab_size'])

    ## ModelInit    
    model = VRAE(params)
    exp_path = "./saved/iwslt_vrnn/"
    model_name = "seq2seq.ckpt"

    ## Session
    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        restore_path = tf.train.latest_checkpoint(exp_path)
        # restore_path = "./saved/seq2seq/seq2seq.ckpt-70002"
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
        # batcher = dataloader._load_data(fpath=test_file)
        test_file = "./corpus/iwslt2015/prepro/test"
        batcher = dataloader.load_data(fpath=test_file)
        for _ in tqdm(range((test_len-1)//args.batch_size+1)):
            try:
                # enc_inp, dec_inp_full, dec_out = next(batcher)
                (enc_inp, _, _, _, _, _), x_enc_inp_oovs, data_oovs, _ = next(batcher)
                # dec_inp = dataloader.update_word_dropout(dec_inp_full)
            except StopIteration:
                print("there are no more examples")
                break
            model.evaluation(sess, enc_inp, trans_file)

    # Evaluation
    eval_log = {}
    for metric in ['bleu','rouge','accuracy','word_accuracy']:
        score = evaluation_utils.evaluate(
            ref_file,
            trans_file,
            metric)
        eval_log[metric] = score
        if metric == "bleu":
            print("  bleu-1, bleu-2, bleu-3, bleu-4: %.5f,  %.5f,  %.5f,  %.5f" % score)
        elif metric == "rouge":
            print("  rouge-1, rouge-2, rouge-l: %.5f,  %.5f,  %.5f" % score)
        else:
            print("  %s: %.5f" % (metric, score))
    
    from measures import selfbleu
    selfbleuobj = selfbleu.SelfBleu(trans_file, 1)
    print("  selfbleu-1", 1-selfbleuobj.get_score())
    eval_log['selfbleu-1'] = selfbleuobj.get_score()
    selfbleuobj = selfbleu.SelfBleu(trans_file, 2)
    print("  selfbleu-2", 1-selfbleuobj.get_score())
    eval_log['selfbleu-2'] = selfbleuobj.get_score()
    selfbleuobj = selfbleu.SelfBleu(trans_file, 3)
    print("  selfbleu-3", 1-selfbleuobj.get_score())
    eval_log['selfbleu-3'] = selfbleuobj.get_score()
    selfbleuobj = selfbleu.SelfBleu(trans_file, 4)
    print("  selfbleu-4", 1-selfbleuobj.get_score())
    eval_log['selfbleu-4'] = selfbleuobj.get_score()

    # Record Log
    dataloader.record_result(eval_log, finpath=test_file, frespaht=trans_file, foutpath=result_file)
    

if __name__ == '__main__':
    print(json.dumps(args.__dict__, indent=4))
    main()