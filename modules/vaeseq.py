import tensorflow as tf
import numpy as np
import sys

from config import args
from modules.modified import ModifiedBasicDecoder, ModifiedBeamSearchDecoder
from modules.basevae import BaseVAE
from modules.transformer import Transformer
from data.data_reddit import START_TOKEN, END_TOKEN, UNK_STRING, PAD_STRING

class VAESEQ:
    def __init__(self, params):
        self.params = params
        self.encoder_model = None
        self.decoder_model = None
        self.transformer_model = None
        self._build_inputs()
        self._init_models(params)
        self._loss_optimizer(params['loss_type'])
        self.print_parameters()


    def print_parameters(self):
        print("print_parameters:")
        for item in tf.global_variables():
            print('%s: %s' % (item.name, item.get_shape()))
    
    def _build_inputs(self):
        with tf.variable_scope('placeholder'):
            # placeholders x
            self.x_enc_inp = tf.placeholder(tf.int32, [None, args.max_len], name="x_enc_inp")
            self.x_dec_inp = tf.placeholder(tf.int32, [None, args.max_len+1], name="x_dec_inp")
            self.x_dec_out = tf.placeholder(tf.int32, [None, args.max_len+1], name="x_dec_out")
            # placeholders y
            self.y_enc_inp = tf.placeholder(tf.int32, [None, args.max_len], name="y_enc_inp")
            self.y_dec_inp = tf.placeholder(tf.int32, [None, args.max_len+1], name="y_dec_inp")
            self.y_dec_out = tf.placeholder(tf.int32, [None, args.max_len+1], name="y_dec_out")
            # train step
            self.global_step = tf.Variable(0, trainable=False)

    def _init_models(self, params):
        # self.global_step = None
        with tf.variable_scope('encodervae'):
            encodervae_inputs = (self.x_enc_inp, self.x_dec_inp, self.x_dec_out, self.global_step)
            self.encoder_model = BaseVAE(params, encodervae_inputs, "encoder")
        with tf.variable_scope('decodervae'):
            decodervae_inputs = (self.y_enc_inp, self.y_dec_inp, self.y_dec_out, self.global_step)
            self.decoder_model = BaseVAE(params, decodervae_inputs, "decoder")
        with tf.variable_scope('transformer'):
            self.transformer = Transformer(self.encoder_model, self.decoder_model, params['graph_type'], self.global_step)
        with tf.variable_scope('decodervae/decoding', reuse=True):
            self.training_rnn_out, self.training_logits = self.decoder_model._decoder_training(self.transformer.predition, reuse=True)
            self.predicted_ids_op = self.decoder_model._decoder_inference(self.transformer.predition)
    
    def _gradient_clipping(self, loss_op):
        params = tf.trainable_variables()
        # print("_gradient_clipping")
        # print(len(params))
        # for item in params:
        #     print('%s: %s' % (item.name, item.get_shape()))
        gradients = tf.gradients(loss_op, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, args.clip_norm)
        # print(len(clipped_gradients))
        # print(_)
        # for item in clipped_gradients[1:]:
        #     print('%s: %s' % (item.name, item.get_shape()))
        # print("_gradient_clipping")
        return clipped_gradients, params

    def _loss_optimizer(self, model_type):
        with tf.variable_scope('merge_loss'):
            mask_fn = lambda l : tf.sequence_mask(l, args.max_dec_len, dtype=tf.float32)
            dec_seq_len = tf.count_nonzero(self.y_dec_out, 1, dtype=tf.int32)
            mask = mask_fn(dec_seq_len) # b x t = 64 x ?
            self.merged_loss_seq =  tf.reduce_sum(tf.contrib.seq2seq.sequence_loss(
                logits = self.training_logits,
                targets = self.y_dec_out,
                weights = mask,
                average_across_timesteps = False,
                average_across_batch = True))
            if model_type == 0:
                self.merged_loss = self.transformer.merged_mse*100000 + self.encoder_model.loss + self.decoder_model.loss
            elif model_type == 1:
                self.merged_loss = self.transformer.wasserstein_loss*1000 + self.encoder_model.loss + self.decoder_model.loss
            elif model_type == 2:
                self.merged_loss = self.merged_loss_seq + self.encoder_model.loss + self.decoder_model.loss

        # with tf.variable_scope('optimizer'):
        #     # self.global_step = tf.Variable(0, trainable=False)
        #     clipped_gradients, params = self._gradient_clipping(self.merged_loss)
        #     self.merged_train_op = tf.train.AdamOptimizer().apply_gradients(
        #         zip(clipped_gradients, params), global_step=self.global_step)
        #     # self.merged_train_op = tf.train.AdamOptimizer(self.merged_loss)
            
        with tf.variable_scope('summary'):
            tf.summary.scalar("trans_loss", self.merged_loss_seq)
            tf.summary.scalar("merged_loss", self.merged_loss)
            tf.summary.histogram("z_predition", self.transformer.predition)
            self.merged_summary_op = tf.summary.merge_all()

    def train_encoder(self, sess, x_enc_inp, x_dec_inp, x_dec_out, y_enc_inp, y_dec_inp, y_dec_out):
        feed_dict = {
            self.x_enc_inp: x_enc_inp,
            self.x_dec_inp: x_dec_inp,
            self.x_dec_out: x_dec_out,
            self.y_enc_inp: y_enc_inp,
            self.y_dec_inp: y_dec_inp,
            self.y_dec_out: y_dec_out
        }
        log = self.encoder_model.train_session(sess, feed_dict)
        return log
        
    def train_decoder(self, sess, x_enc_inp, x_dec_inp, x_dec_out, y_enc_inp, y_dec_inp, y_dec_out):
        feed_dict = {
            self.x_enc_inp: x_enc_inp,
            self.x_dec_inp: x_dec_inp,
            self.x_dec_out: x_dec_out,
            self.y_enc_inp: y_enc_inp,
            self.y_dec_inp: y_dec_inp,
            self.y_dec_out: y_dec_out
        }
        log = self.decoder_model.train_session(sess, feed_dict)
        return log
    
    def train_transformer(self, sess, x_enc_inp, x_dec_inp, x_dec_out, y_enc_inp, y_dec_inp, y_dec_out):
        feed_dict = {
            self.x_enc_inp: x_enc_inp,
            self.x_dec_inp: x_dec_inp,
            self.x_dec_out: x_dec_out,
            self.y_enc_inp: y_enc_inp,
            self.y_dec_inp: y_dec_inp,
            self.y_dec_out: y_dec_out
        }
        log = self.transformer.train_session(sess, feed_dict)
        return log
        
    # def merged_train(self, sess, x_enc_inp, x_dec_inp, x_dec_out, y_enc_inp, y_dec_inp, y_dec_out):
    #     log = self.transformer.merged_train_session(sess, self.encoder_model, self.decoder_model,
    #             x_enc_inp, x_dec_inp, x_dec_out, y_enc_inp, y_dec_inp, y_dec_out)
    #     return log

    def merged_seq_train(self, sess, x_enc_inp, x_dec_inp, x_dec_out, y_enc_inp, y_dec_inp, y_dec_out):
        feed_dict = {
            self.x_enc_inp: x_enc_inp,
            self.x_dec_inp: x_dec_inp,
            self.x_dec_out: x_dec_out,
            self.y_enc_inp: y_enc_inp,
            self.y_dec_inp: y_dec_inp,
            self.y_dec_out: y_dec_out
        }
        _, summaries, loss, trans_loss, encoder_loss, decoder_loss, step = sess.run(
            [self.merged_train_op, self.merged_summary_op, self.merged_loss, self.merged_loss_seq, self.encoder_model.loss, self.decoder_model.loss, self.global_step],
                feed_dict)
        return {'summaries': summaries, 'merged_loss': loss, 'trans_loss': trans_loss, 
            'encoder_loss': encoder_loss, 'decoder_loss': decoder_loss, 'step': step}


    def show_encoder(self, sess, x, y, LOGGER):
        # self.encoder_model.generate(sess)
        infos = self.encoder_model.reconstruct(sess, x, y)
        # self.encoder_model.customized_reconstruct(sess, 'i love this film and i think it is one of the best films')
        # self.encoder_model.customized_reconstruct(sess, 'this movie is a waste of time and there is no point to watch it') 
        LOGGER.write(infos)
        print(infos.strip())

    def show_decoder(self, sess, x, y, LOGGER):
        # self.decoder_model.generate(sess)
        infos = self.decoder_model.reconstruct(sess, x, y)
        # self.decoder_model.customized_reconstruct(sess, 'i love this film and i think it is one of the best films')
        # self.decoder_model.customized_reconstruct(sess, 'this movie is a waste of time and there is no point to watch it')
        LOGGER.write(infos)
        print(infos.strip())

    def show_sample(self, sess, x, y, LOGGER):
        infos = self.transformer.sample_test(sess, x, y, self.encoder_model, self.decoder_model, self.predicted_ids_op)
        LOGGER.write(infos)
        print(infos.strip())

    def evaluation_encoder_vae(self, sess, enc_inp, outputfile):
        self.encoder_model.evaluation(sess, enc_inp, outputfile)

    def evaluation_decoder_vae(self, sess, enc_inp, outputfile):
        self.decoder_model.evaluation(sess, enc_inp, outputfile)
    
    # def evaluation(self, sess, enc_inp, outputfile):
    #     idx2word = self.params['idx2word']
    #     batch_size, predicted_decoder_z = sess.run([self.encoder_model._batch_size, self.transformer.predition], {self.encoder_model.enc_inp:enc_inp})
    #     predicted_ids_lt = sess.run(self.decoder_model.predicted_ids, 
    #         {self.decoder_model._batch_size: batch_size, self.decoder_model.z: predicted_decoder_z,
    #             self.decoder_model.enc_seq_len: [args.max_len]})
    #     for predicted_ids in predicted_ids_lt:
    #         with open(outputfile, "a") as f:
    #             f.write('%s\n' % ' '.join([idx2word[idx] for idx in predicted_ids]))

    def evaluation(self, sess, enc_inp, outputfile):
        idx2word = self.params['idx2word']
        #### method - I
        # batch_size, trans_input = sess.run([self.encoder_model._batch_size, self.encoder_model.z], {self.x_enc_inp:enc_inp})
        # predicted_decoder_z = sess.run(self.transformer.predition, {self.transformer.input:trans_input})
        #### method - I.2.0
        batch_size, trans_input_mean, trans_input_logvar = sess.run([self.encoder_model._batch_size, self.encoder_model.z_mean, self.encoder_model.z_logvar], {self.x_enc_inp:enc_inp})
        predicted_decoder_z = sess.run(self.transformer.predition, {self.transformer.input_mean:trans_input_mean, self.transformer.input_logvar:trans_input_logvar})
        # print("========================")
        # print(trans_input)
        # print("------------------------")
        # print(predicted_decoder_z)
        # print("========================")
        predicted_ids_lt = sess.run(self.predicted_ids_op, 
            {self.decoder_model._batch_size: batch_size, self.decoder_model.z: predicted_decoder_z,
                self.decoder_model.enc_seq_len: [args.max_len]})
        #### method - II
        # batch_size = sess.run(self.encoder_model._batch_size, {self.x_enc_inp:enc_inp})
        # predicted_ids_lt = sess.run(self.predicted_ids_op, 
        #     {self.decoder_model._batch_size: batch_size, self.x_enc_inp: enc_inp, self.y_enc_inp: enc_inp, self.decoder_model.enc_seq_len: [args.max_len]})
        for predicted_ids in predicted_ids_lt:
            with open(outputfile, "a") as f:
                result = ' '.join([idx2word[idx] for idx in predicted_ids])
                end_index = result.find(" </S> ")
                if end_index != -1:
                    result = result[:end_index]
                f.write('%s\n' % result)
                # f.write('%s\n' % ' '.join([idx2word[idx] for idx in predicted_ids]))