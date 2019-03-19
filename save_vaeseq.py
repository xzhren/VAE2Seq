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
        
    exp_path = "./saved/"+args.exp+"/"
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    model_name = args.model_name
    train_data_len = 3384185
    train_data_path = args.train_data
    EPOCH_STEPS = (train_data_len-1)//args.batch_size+1
    print(args)

    ## DataLoader
    dataloader = REDDIT(batch_size=args.batch_size, vocab_limit=args.vocab_limit, max_input_len=args.enc_max_len, max_output_len=args.dec_max_len)
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
    # variables = [v for v in variables if not (v.name.startswith("encodervae/optimizer/") or v.name.startswith("decodervae/optimizer/"))]
    for v in variables:
        print(type(v.name), v.name)
    print(len(variables))
    # end load
    saver = tf.train.Saver(variables)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        restore_path = tf.train.latest_checkpoint(exp_path)
        saver.restore(sess, restore_path)
        # save to folder
        saver.save(sess, "saved/bivae_raw/"+"vrae.ckpt", global_step=0)


if __name__ == '__main__':
    print(json.dumps(args.__dict__, indent=4))
    main()