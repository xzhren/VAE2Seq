from __future__ import print_function
from data_reddit import REDDIT
from model_vae import VRAE
from config import args
import json
import tensorflow as tf
from tqdm import tqdm

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
    args.display_info_step = 10000
    print(args)
    model = VRAE(params)
    saver = tf.train.Saver()

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter("./saved/")
    saver.restore(sess, './saved/vrae.ckpt')

    train_data_len = 3384185
    for epoch in range(args.num_epoch):
        # dataloader.update_word_dropout()
        # print("\nWord Dropout")
        # dataloader.shuffle()
        # print("Data Shuffled", end='\n\n')

        step = -1
        while True:
            try:
                # enc_inp, dec_inp_full, dec_out = next(dataloader.data_loader)
                enc_inp, dec_inp_full, dec_out = dataloader.next_batch()
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
                model.generate(sess)
                model.reconstruct(sess, enc_inp[-1], dec_inp[-1])
                model.customized_reconstruct(sess, 'i love this film and i think it is one of the best films')
                model.customized_reconstruct(sess, 'this movie is a waste of time and there is no point to watch it')
                save_path = saver.save(sess, './saved/vrae.ckpt', global_step=train_step)
                print("Model saved in file: %s" % save_path)

        model.generate(sess)
        model.reconstruct(sess, enc_inp[-1], dec_inp[-1])
        model.customized_reconstruct(sess, 'i love this film and i think it is one of the best films')
        model.customized_reconstruct(sess, 'this movie is a waste of time and there is no point to watch it')
        
        save_path = saver.save(sess, './saved/vrae.ckpt')
        print("Model saved in file: %s" % save_path)


if __name__ == '__main__':
    print(json.dumps(args.__dict__, indent=4))
    main()