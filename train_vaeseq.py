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
    LOGGER = open(log_path, "a")

    ## Session
    # load some parameters
    variables = tf.contrib.framework.get_variables_to_restore()
    # print(len(variables), end=",")
    # variables = [v for v in variables if not v.name.startswith("optimizer/transformer/trans_mlp/")]
    # for v in variables_to_resotre:
    #     print(type(v.name), v.name)
    print(len(variables))
    # end load
    saver = tf.train.Saver(variables)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter(exp_path, sess.graph)
    # tf.train.write_graph(sess.graph, './saved/vaeseq/', 'train.pbtxt')
    keep_on_train_flag = False
    restore_path = tf.train.latest_checkpoint(exp_path)
    if restore_path:
        keep_on_train_flag = True
        saver.restore(sess, restore_path)
        saver = tf.train.Saver() # new saver
        last_train_step = int(restore_path.split("-")[-1]) % EPOCH_STEPS
        print("Model restore from file: %s, last train step: %d" % (restore_path, last_train_step))
        LOGGER.write("Model restore from file: %s, last train step: %d\n" % (restore_path, last_train_step))

    # Train Mode
    x_log, y_log, t_log, log = None, None, None, None
    for epoch in range(args.num_epoch):
        batcher = dataloader.load_data(fpath=train_data_path)
        for step in tqdm(range(EPOCH_STEPS)):
            if keep_on_train_flag and step <= last_train_step: continue
            if keep_on_train_flag and step == (last_train_step+1): keep_on_train_flag=False

            # get batch data
            try:
                x_enc_inp, x_dec_inp_full, x_dec_out, y_enc_inp, y_dec_inp_full, y_dec_out = next(batcher)
                x_dec_inp = dataloader.update_word_dropout(x_dec_inp_full)
                y_dec_inp = dataloader.update_word_dropout(y_dec_inp_full)
            except StopIteration:
                print("there are no more examples")
                break
                
            # x_log = model.train_encoder(sess, x_enc_inp, x_dec_inp, x_dec_out, y_enc_inp, y_dec_inp, y_dec_out)
            # y_log = model.train_decoder(sess, x_enc_inp, x_dec_inp, x_dec_out, y_enc_inp, y_dec_inp, y_dec_out)
            t_log = model.train_transformer(sess, x_enc_inp, x_dec_inp, x_dec_out, y_enc_inp, y_dec_inp, y_dec_out)
            # log = model.merged_train(sess, x_enc_inp, x_dec_inp, x_dec_out, y_enc_inp, y_dec_inp, y_dec_out)
            # log = model.merged_seq_train(sess, x_enc_inp, x_dec_inp, x_dec_out, y_enc_inp, y_dec_inp, y_dec_out)
            # model.show_parameters(sess)

            # get the summaries and iteration number so we can write summaries to tensorboard
            train_step = summary_flush(x_log, y_log, t_log, log, summary_writer)

            if step % args.display_loss_step == 0:
                print("Step %d | [%d/%d] | [%d/%d]" % (train_step, epoch+1, args.num_epoch, step, train_data_len//args.batch_size), end='')
                LOGGER.write("Step %d | [%d/%d] | [%d/%d]" % (train_step, epoch+1, args.num_epoch, step, train_data_len//args.batch_size))
                show_loss(x_log, y_log, t_log, log, LOGGER)
        
            if step % args.display_info_step == 0 and step != 0:
                args.training = False
                save_path = saver.save(sess, exp_path+model_name, global_step=train_step)
                print("Model saved in file: %s" % save_path)
                print("============= Show Encoder ===============")
                model.show_encoder(sess, x_enc_inp[-1], x_dec_inp[-1], LOGGER)
                print("============= Show Decoder ===============")
                model.show_decoder(sess, y_enc_inp[-1], y_dec_inp[-1], LOGGER)
                print("============= Show Sample ===============")
                for i in range(3):
                    model.show_sample(sess, x_enc_inp[i], y_dec_out[i], LOGGER)
                LOGGER.flush()
                args.training = True
                
        save_path = saver.save(sess, exp_path+model_name, global_step=train_step)
        print("Model saved in file: %s" % save_path)
        model.show_encoder(sess, x_enc_inp[-1], x_dec_inp[-1], LOGGER)
        model.show_decoder(sess, y_enc_inp[-1], y_dec_inp[-1], LOGGER)
        model.show_sample(sess, x_enc_inp[-1], y_dec_out[-1], LOGGER)


if __name__ == '__main__':
    print(json.dumps(args.__dict__, indent=4))
    main()