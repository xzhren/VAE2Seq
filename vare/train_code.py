from __future__ import print_function

from data_code import CODE, TEST_DATA_SIZE, TRADIN_DATA_SIZE
from model_vae import VRAE
from config import args

import json
import tensorflow as tf
from tqdm import tqdm
import os

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    args.enc_max_len = 50
    args.dec_max_len = 30
    args.vocab_limit = 10000
    dataloader = CODE(batch_size=args.batch_size, vocab_limit=args.vocab_limit, max_input_len=args.enc_max_len, max_output_len=args.dec_max_len)

    params = {
        'vocab_size': len(dataloader.idx2word),
        'word2idx': dataloader.word2idx,
        'idx2word': dataloader.idx2word,
        'token2idx': dataloader.token2idx,
        'idx2token': dataloader.idx2token}
    print('Vocab Size:', params['vocab_size'])
    args.max_len = args.enc_max_len
    # args.batch_size = 64
    args.max_dec_len = args.dec_max_len
    args.display_info_step = 10000
    print(args)
    model = VRAE(params)
    saver = tf.train.Saver()
    exp_path = "./saved/"+args.exp+"/"
    model_name = "vaeseq.ckpt"

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    train_data_len = TRADIN_DATA_SIZE
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
        batcher = dataloader.load_train_data()

        step = -1
        while True:
            try:
                enc_inp, _, _, _, dec_inp_full, dec_out = next(batcher)
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
                print(" | nll_loss:%.1f | kl_w:%.3f | kl_loss:%.2f" % (log['nll_loss'], log['kl_w'], log['kl_loss']))
        
            if step % args.display_info_step == 0:
                # model.generate(sess)
                model.reconstruct(sess, enc_inp[-1], dec_inp[-1])
                # model.customized_reconstruct(sess, 'i love this film and i think it is one of the best films')
                # model.customized_reconstruct(sess, 'this movie is a waste of time and there is no point to watch it')
                save_path = saver.save(sess, exp_path+model_name, global_step=train_step)
                print("Model saved in file: %s" % save_path)

        # model.generate(sess)
        model.reconstruct(sess, enc_inp[-1], dec_inp[-1])
        # model.customized_reconstruct(sess, 'i love this film and i think it is one of the best films')
        # model.customized_reconstruct(sess, 'this movie is a waste of time and there is no point to watch it')
        
        save_path = saver.save(sess, exp_path+model_name, global_step=train_step)
        print("Model saved in file: %s" % save_path)


if __name__ == '__main__':
    print(json.dumps(args.__dict__, indent=4))
    main()