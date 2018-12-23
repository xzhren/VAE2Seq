from __future__ import print_function
from modules.data_reddit import REDDIT
from modules.vaeseq import VAESEQ
from config import args
import json
import tensorflow as tf
from tqdm import tqdm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    ## Parameters
    args.max_len = 150
    args.batch_size = 64
    args.max_dec_len = 151
    args.display_info_step = 10000
    print(args)

    ## DataLoader
    dataloader = REDDIT(batch_size=64, vocab_limit=35000, max_input_len=150, max_output_len=150)
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

    summary_writer = tf.summary.FileWriter("./saved/vaeseq/")
    # saver.restore(sess, './saved/vrae.ckpt')

    # Train Mode
    train_data_len = 3384185
    train_data_path = "./corpus/reddit/train.txt"
    batcher = dataloader.load_data(fpath=train_data_path)
    for epoch in range(args.num_epoch):
        for step in tqdm(range((train_data_len-1)//args.batch_size+1)):
            # get batch data
            try:
                x_enc_inp, x_dec_inp_full, x_dec_out, y_enc_inp, y_dec_inp_full, y_dec_out = next(batcher)
                x_dec_inp = dataloader.update_word_dropout(x_dec_inp_full)
                y_dec_inp = dataloader.update_word_dropout(y_dec_inp_full)
            except StopIteration:
                print("there are no more examples")
                break
                
            # x_log = model.train_encoder(sess, x_enc_inp, x_dec_inp, x_dec_out)
            # y_log = model.train_encoder(sess, y_enc_inp, y_dec_inp, y_dec_out)
            # t_log = model.train_transformer(sess, x_enc_inp, x_dec_inp, x_dec_out, y_enc_inp, y_dec_inp, y_dec_out)
            log = model.merged_train(sess, x_enc_inp, x_dec_inp, x_dec_out, y_enc_inp, y_dec_inp, y_dec_out)

            # get the summaries and iteration number so we can write summaries to tensorboard
            summaries, train_step = log['summaries'], log['step']
            summary_writer.add_summary(summaries, train_step) # write the summaries
            
            if train_step % 100 == 0: # flush the summary writer every so often
                summary_writer.flush()

            if step % args.display_loss_step == 0:
                print("Step %d | [%d/%d] | [%d/%d]" % (log['step'], epoch+1, args.num_epoch, step, train_data_len//args.batch_size), end='')
                print(" | nll_loss:%.1f | kl_w:%.3f | kl_loss:%.2f" % (log['merged_loss'], log['merged_loss'], log['merged_loss']))
                # print(" | nll_loss:%.1f | kl_w:%.3f | kl_loss:%.2f" % (log['nll_loss'], log['kl_w'], log['kl_loss']))
        
            if step % args.display_info_step == 0 and step != 0:
                # model.show_encoder(sess, x_enc_inp[-1], x_dec_inp[-1])
                # model.show_decoder(sess, y_enc_inp[-1], y_dec_inp[-1])
                model.show_sample(sess, x_enc_inp[-1], y_dec_inp[-1])
                save_path = saver.save(sess, './saved/vaeseq/vrae.ckpt', global_step=train_step)
                print("Model saved in file: %s" % save_path)
                
        model.show_encoder(sess, x_enc_inp[-1], x_dec_inp[-1])
        model.show_decoder(sess, y_enc_inp[-1], y_dec_inp[-1])
        model.show_sample(sess, x_enc_inp[-1], y_dec_inp[-1])
        save_path = saver.save(sess, './saved/vaeseq/vrae.ckpt', global_step=train_step)
        print("Model saved in file: %s" % save_path)


if __name__ == '__main__':
    print(json.dumps(args.__dict__, indent=4))
    main()