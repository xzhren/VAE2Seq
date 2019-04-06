from __future__ import print_function

# from data_reddit import REDDIT
from model import VRAE
from config import args

import json
import tensorflow as tf
from tqdm import tqdm
import os

import sys
sys.path.append(".")
from data.data_iwslt import IWSLT

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    args.max_len = 100
    args.batch_size = 64
    args.max_dec_len = 100
    args.display_info_step = 1000
    args.isPointer = False
    args.vocab_limit = 35000
    train_data_len = 209940
    args.diff_input = True
    print(args)

    dataloader = IWSLT(batch_size=args.batch_size, vocab_limit=args.vocab_limit, max_input_len=args.max_len, max_output_len=args.max_dec_len)
    params = {
        'vocab_size_encoder': len(dataloader.idx2token),
        'vocab_size': len(dataloader.word2idx),
        'word2idx': dataloader.word2idx,
        'idx2word': dataloader.idx2word,
        'idx2token': dataloader.idx2token}
    print('Vocab Size:', params['vocab_size'])

    model = VRAE(params)
    saver = tf.train.Saver()
    exp_path = "./saved/iwslt_copynet/"
    model_name = "seq2seq.ckpt"

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    EPOCH_STEPS = (train_data_len-1)//args.batch_size+1

    summary_writer = tf.summary.FileWriter(exp_path, sess.graph)
    restore_path = tf.train.latest_checkpoint(exp_path)
    if restore_path:
        saver.restore(sess, restore_path)
        last_train_step = int(restore_path.split("-")[-1]) % EPOCH_STEPS
        print("Model restore from file: %s, last train step: %d" % (restore_path, last_train_step))
    # summary_writer = tf.summary.FileWriter(exp_path)
    # saver.restore(sess, exp_path+model_name)

    for epoch in range(args.num_epoch):
        # dataloader.update_word_dropout()
        # print("\nWord Dropout")
        # dataloader.shuffle()
        # print("Data Shuffled", end='\n\n')
        batcher = dataloader.load_data()

        step = -1
        while True:
            try:
                # enc_inp, dec_inp_full, dec_out = next(dataloader.data_loader)
                (x_enc_inp, x_dec_inp_full, x_dec_out, y_enc_inp, y_dec_inp_full, y_dec_out), x_enc_inp_oovs, data_oovs, _ = next(batcher)
                # enc_inp, dec_inp_full, dec_out = dataloader.next_batch()
                enc_inp, dec_inp_full, dec_out = x_enc_inp, y_dec_inp_full, y_dec_out
                dec_inp = dataloader.update_word_dropout(dec_inp_full)
                step += 1
            except StopIteration:
                print("there are no more examples")
                break
            # print(step, "enc_inp.shape:", enc_inp.shape)
            # print(step, "dec_inp_full.shape:", dec_inp_full.shape)
            # print(step, "dec_out.shape:", dec_out.shape)

            log = model.train_session(sess, enc_inp, dec_inp, dec_out)

            # get the summaries and iteration number so we can write summaries to tensorboard
            summaries, train_step = log['summaries'], log['step']
            summary_writer.add_summary(summaries, train_step) # write the summaries
            if train_step % 100 == 0: # flush the summary writer every so often
                summary_writer.flush()

            if step % args.display_loss_step == 0:
                print("Step %d | [%d/%d] | [%d/%d]" % (log['step'], epoch+1, args.num_epoch, step, train_data_len//args.batch_size), end='')
                print(" | loss:%.3f" % (log['loss']))
        
            if step % args.display_info_step == 0:
                model.reconstruct(sess, enc_inp[-1], dec_out[-1])
                save_path = saver.save(sess, exp_path+model_name, global_step=train_step)
                print("Model saved in file: %s" % save_path)

        model.reconstruct(sess, enc_inp[-1], dec_out[-1])
        save_path = saver.save(sess, exp_path+model_name, global_step=train_step)
        print("Model saved in file: %s" % save_path)


if __name__ == '__main__':
    print(json.dumps(args.__dict__, indent=4))
    main()