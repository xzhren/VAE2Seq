from __future__ import print_function
from data_reddit import REDDIT
from model_vae import VRAE
from config import args
import json
import tensorflow as tf
from tqdm import tqdm
from measures import evaluation_utils

def main():
    dataloader = REDDIT(batch_size=64, vocab_limit=35000, max_input_len=150, max_output_len=150)
    params = {
        'vocab_size': len(dataloader.word2idx),
        'word2idx': dataloader.word2idx,
        'idx2word': dataloader.idx2word,}
    print('Vocab Size:', params['vocab_size'])
    args.max_len = 150
    args.batch_size = 64
    args.max_dec_len = 151
    # args.display_info_step = 10000
    print(args)
    model = VRAE(params)
    saver = tf.train.Saver()

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    saver.restore(sess, './saved/vrae.ckpt')
    print("[RESTORE MODEL]")

    # Parpear Dir
    ref_file = "./saved/test.input.txt"
    trans_file = "./saved/test.output.txt"
    result_file = "./saved/test.result.txt"
    test_file = "./corpus/reddit/test.txt"

    # Test Dir
    dataloader.trans_in_ref(finpath=test_file, foutpath=ref_file)
    with open(trans_file, "w") as f:
        f.write("")
    print("[PAEPEAR DATASET]")

    # Test DataSet
    test_len = 20000
    batcher = dataloader._load_data(fpath=test_file)
    for _ in tqdm(range((test_len-1)//args.batch_size+1)):
        try:
            enc_inp, _, _ = next(batcher)
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
        print("  %s: %.1f" % (metric, score))

    # Record Log
    dataloader.record_result(eval_log, finpath=test_file, frespaht=trans_file, foutpath=result_file)
    

if __name__ == '__main__':
    print(json.dumps(args.__dict__, indent=4))
    main()