import json
import tensorflow as tf
from tqdm import tqdm
import os

import sys
sys.path.append(".")

from data.data_reddit import REDDIT
from modules.vaeseq import VAESEQ
from config import args
from measures import evaluation_utils

def main():
    ## CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    ## Parameters
    if args.exp == "NONE":
        args.exp = args.graph_type
        
    exp_path = "./saved/"+args.exp+"/"
    args.data_len = 20000
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

    ## Session
    # load some parameters
    variables = tf.contrib.framework.get_variables_to_restore()
    print(len(variables), end=",")
    variables = [v for v in variables if "optimizer" not in v.name]
    # for v in variables_to_resotre:
    #     print(type(v.name), v.name)
    print(len(variables))
    # end load
    saver = tf.train.Saver(variables)
    # saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        restore_path = tf.train.latest_checkpoint(exp_path)
        saver.restore(sess, restore_path)
        saver = tf.train.Saver() # new saver
        print("Model restore from file: %s" % (restore_path))
        
        # Parpear Dir
        ref_file = exp_path+"test.input.txt"
        trans_file = exp_path+"test.output.txt"
        result_file = exp_path+"test."+restore_path.split("-")[-1]+".decoder.result.txt"
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
                x_enc_inp, _, _, y_enc_inp, _, _ = next(batcher)
                # dec_inp = dataloader.update_word_dropout(dec_inp_full)
            except StopIteration:
                print("there are no more examples")
                break
            # print("x_enc_inp:", x_enc_inp)
            # print("y_enc_inp:", y_enc_inp)
            # model.decoder_model.generate(sess)
            # model.decoder_model.generate(sess)
            # model.decoder_model.generate(sess)
            # model.decoder_model.generate(sess)
            # model.decoder_model.generate(sess)
            print("-----------------")
            model_test = model.decoder_model
            predict_z_one = model_test.point_reconstruct(sess, "have fun")
            predict_z_two = model_test.point_reconstruct(sess, "thank you very much")
            print("-----------------")
            beta = 0.8
            model_test.generate_byz(sess, beta*predict_z_one+(1-beta)*predict_z_two)
            beta = 0.6
            model_test.generate_byz(sess, beta*predict_z_one+(1-beta)*predict_z_two)
            beta = 0.4
            model_test.generate_byz(sess, beta*predict_z_one+(1-beta)*predict_z_two)
            beta = 0.2
            model_test.generate_byz(sess, beta*predict_z_one+(1-beta)*predict_z_two)
            break


if __name__ == '__main__':
    print(json.dumps(args.__dict__, indent=4))
    main()