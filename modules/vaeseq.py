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

    def _init_models(self, params):
        with tf.variable_scope('encodervae'):
            encodervae_inputs = (self.x_enc_inp, self.x_dec_inp, self.x_dec_out)
            self.encoder_model = BaseVAE(params, encodervae_inputs, "encoder")
        with tf.variable_scope('decoderrvae'):
            encodervae_inputs = (self.y_enc_inp, self.y_dec_inp, self.y_dec_out)
            self.decoder_model = BaseVAE(params, encodervae_inputs, "decoder")
        with tf.variable_scope('transformer'):
            self.transformer = Transformer(self.encoder_model, self.decoder_model)
        with tf.variable_scope('decoderrvae/decoding', reuse=True):
            self.predicted_ids_op = self.decoder_model._decoder_inference(self.transformer.predition)

    def train_encoder(self, sess, enc_inp, dec_inp, dec_out):
        log = self.encoder_model.train_session(sess, enc_inp, dec_inp, dec_out)
        return log
        
    def train_decoder(self, sess, enc_inp, dec_inp, dec_out):
        log = self.decoder_model.train_session(sess, enc_inp, dec_inp, dec_out)
        return log
    
    def train_transformer(self, sess, x_enc_inp, x_dec_inp, x_dec_out, y_enc_inp, y_dec_inp, y_dec_out):
        log = self.transformer.train_session(sess, self.encoder_model, self.decoder_model,
                x_enc_inp, x_dec_inp, x_dec_out, y_enc_inp, y_dec_inp, y_dec_out)
        return log
        
    def merged_train(self, sess, x_enc_inp, x_dec_inp, x_dec_out, y_enc_inp, y_dec_inp, y_dec_out):
        log = self.transformer.merged_train_session(sess, self.encoder_model, self.decoder_model,
                x_enc_inp, x_dec_inp, x_dec_out, y_enc_inp, y_dec_inp, y_dec_out)
        return log

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
    
    def evaluation(self, sess, enc_inp, outputfile):
        idx2word = self.params['idx2word']
        batch_size, predicted_decoder_z = sess.run([self.encoder_model._batch_size, self.transformer.predition], {self.encoder_model.enc_inp:enc_inp})
        predicted_ids_lt = sess.run(self.decoder_model.predicted_ids, 
            {self.decoder_model._batch_size: batch_size, self.decoder_model.z: predicted_decoder_z,
                self.decoder_model.enc_seq_len: [args.max_len]})
        for predicted_ids in predicted_ids_lt:
            with open(outputfile, "a") as f:
                f.write('%s\n' % ' '.join([idx2word[idx] for idx in predicted_ids]))
