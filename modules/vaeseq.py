from config import args
from modified import ModifiedBasicDecoder, ModifiedBeamSearchDecoder
from modules.basevae import BaseVAE
from modules.transformer import Transformer

import tensorflow as tf
import numpy as np
import sys

from data_reddit import START_TOKEN, END_TOKEN, UNK_STRING, PAD_STRING

class VAESEQ:
    def __init__(self, params):
        self.params = params
        self.encoder_model = None
        self.decoder_model = None
        self.transformer_model = None
        self._init_models(params)
    
    def _init_models(self, params):
        with tf.variable_scope('encodervae'):
            self.encoder_model = BaseVAE(params, "encoder")
        with tf.variable_scope('decoderrvae'):
            self.decoder_model = BaseVAE(params, "decoder")
        with tf.variable_scope('transformer'):
            self.transformer = Transformer(self.encoder_model, self.decoder_model)
        with tf.variable_scope('decoderrvae'):
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

    def show_encoder(self, sess, x, y):
        self.encoder_model.generate(sess)
        self.encoder_model.reconstruct(sess, x, y)
        self.encoder_model.customized_reconstruct(sess, 'i love this film and i think it is one of the best films')
        self.encoder_model.customized_reconstruct(sess, 'this movie is a waste of time and there is no point to watch it') 

    def show_decoder(self, sess, x, y):
        self.decoder_model.generate(sess)
        self.decoder_model.reconstruct(sess, x, y)
        self.decoder_model.customized_reconstruct(sess, 'i love this film and i think it is one of the best films')
        self.decoder_model.customized_reconstruct(sess, 'this movie is a waste of time and there is no point to watch it')

    def show_sample(self, sess, x, y):
        self.transformer.sample_test(sess, x, y, self.encoder_model, self.decoder_model, self.predicted_ids_op)