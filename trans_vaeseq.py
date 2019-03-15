from __future__ import print_function

import json
import tensorflow as tf
from tqdm import tqdm
import os

from data.data_reddit import REDDIT
from modules.vaeseq import VAESEQ
from utils.train_utils import show_loss
from utils.train_utils import summary_flush
from config import args

def main():
    ## CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    ## Parameters
    if args.exp == "NONE":
        args.exp = args.graph_type
    args.max_dec_len = args.max_len+1
    exp_path = "./saved/"+args.exp+"/"
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    model_name = args.model_name
    train_data_len = 3384185
    train_data_path = args.train_data
    EPOCH_STEPS = (train_data_len-1)//args.batch_size+1
    print(args)

    ## DataLoader
    dataloader = REDDIT(batch_size=args.batch_size, vocab_limit=args.vocab_limit, max_input_len=args.max_len, max_output_len=args.max_len)
    params = {
        'vocab_size': len(dataloader.word2idx),
        'word2idx': dataloader.word2idx,
        'idx2word': dataloader.idx2word,
        'loss_type': args.loss_type,
        'graph_type': args.graph_type}
    print('Vocab Size:', params['vocab_size'])

    ## ModelInit    
    model = VAESEQ(params)
    log_path = exp_path+"log.txt"
    # LOGGER = open(log_path, "a")

    ## Session
    # load some parameters
    variables = tf.contrib.framework.get_variables_to_restore()
    print(len(variables), end=",")
    variables = [v for v in variables if v.name.startswith("encodervae/") or v.name.startswith("decodervae/")]
    variables = [v for v in variables if not (v.name.startswith("encodervae/optimizer/") or v.name.startswith("decodervae/optimizer/"))]
    for v in variables:
        print(type(v.name), v.name)
    print(len(variables))
    # end load
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        # summary_writer = tf.summary.FileWriter(exp_path, sess.graph)
        # tf.train.write_graph(sess.graph, './saved/vaeseq/', 'train.pbtxt')

        saver = tf.train.Saver() # new saver
        restore_path = tf.train.latest_checkpoint(exp_path)
        if restore_path:
            saver.restore(sess, restore_path)
    
        saver = tf.train.Saver(variables)    
        restore_path = tf.train.latest_checkpoint("saved/bivae_raw/")
        saver.restore(sess, restore_path)
        
        saver = tf.train.Saver() # new saver
        restore_path = tf.train.latest_checkpoint(exp_path)
        if restore_path:
            last_train_step = int(restore_path.split("-")[-1])
            print("Model restore from file: %s, last train step: %d" % (restore_path, last_train_step))
        else:
            last_train_step = 0
        saver.save(sess, exp_path+model_name, global_step=last_train_step+1)


if __name__ == '__main__':
    print(json.dumps(args.__dict__, indent=4))
    main()