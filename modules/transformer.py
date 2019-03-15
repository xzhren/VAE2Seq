import tensorflow as tf
import numpy as np

from config import args
from data.data_reddit import PAD_TOKEN
from modules.wasserstein_utils import wasserstein_loss
from modules.transformer_utils import *

class Transformer:
    def __init__(self, encoder, decoder, graph_type, global_step):
        self.input_mean = encoder.z_mean
        self.output_mean = decoder.z_mean
        self.input_logvar = encoder.z_logvar
        self.output_logvar = decoder.z_logvar
        self.input = encoder.z
        self.output = decoder.z
        self.global_step = global_step

        self.encoder_class_num = 300
        self.decoder_class_num = 300

        self._build_graph(encoder.loss, decoder.loss, graph_type)
        self._init_summary()

    def _build_graph(self, encoder_loss, decoder_loss, graph_type):
        if graph_type == 'mlp_dist':
            self.predition_mean, self.predition_logvar, self.predition = trans_dist_mlp(args.latent_size, self.input_mean, self.input_logvar)
        elif graph_type == 'mlp_vector':
            self.predition = trans_vector_mlp(args.latent_size, self.input)
        elif graph_type == "embed_dist":
            self.predition_mean, self.predition_logvar, self.predition = trans_dist_embedding(args.latent_size, self.encoder_class_num, self.decoder_class_num, self.input_mean, self.input_logvar)
        elif graph_type == "embed_vector":
            self.predition = trans_vector_embedding(args.latent_size, self.encoder_class_num, self.decoder_class_num, self.input)
        elif graph_type == "trans_dist":
            self.predition_mean, self.predition_logvar, self.predition = trans_dist_transformer(args.latent_size, self.input_mean, self.input_logvar, args.training)
        else:
            print("ERROR graph type, should in [mlp_dist, mlp_vector, embed_dist, embed_vector]")
            import sys
            sys.exit(0)

        with tf.variable_scope('loss'):
            self.loss = tf.losses.mean_squared_error(self.predition, self.output)
            self.wasserstein_loss = wasserstein_loss(self.predition, self.output)
            self.merged_mse = self.loss
            if graph_type == 'mlp_dist' or graph_type == "embed_dist":
                self.loss_mean = tf.losses.mean_squared_error(self.predition_mean, self.output_mean)
                self.loss_logvar = tf.losses.mean_squared_error(self.predition_logvar, self.output_logvar)
                # self.merged_loss = (self.loss_mean+self.loss_logvar+self.loss)*1000 + encoder_loss + decoder_loss
                self.merged_mse = (self.loss_mean+self.loss_logvar+self.loss)


    def build_loss(self, loss_op):
        with tf.variable_scope('optimizer'):
            # self.global_step = tf.Variable(0, trainable=False)
            # loss_op = self.merged_mse

            clipped_gradients, params = self._gradient_clipping(loss_op)
            print("======== [transformer params]", len(clipped_gradients))
            clipped_gradients_, params_ = [], []
            for k, v in zip(clipped_gradients, params):
                if not v.name.startswith("encodervae/") and not v.name.startswith("decodervae/"):
                    clipped_gradients_.append(k)
                    params_.append(v)
            clipped_gradients, params = clipped_gradients_, params_
            for k, v in zip(clipped_gradients, params):
                print(v.name)
            print("======== [end]", len(clipped_gradients))
            self.train_op = tf.train.AdamOptimizer().apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step)
            # self.train_op = tf.train.AdamOptimizer(loss_op)

            # optimizer = tf.train.AdamOptimizer(1e-3)
            # # output_vars = tf.get_collection(tf.GraphKyes.TRAINABLE_VARIABLES)
            # output_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            # print("transformer vars:")
            # print(len(output_vars))
            # # print(len(output_vars), end=",")
            # output_vars = [v for v in output_vars if not v.name.startswith("encodervae/") and not v.name.startswith("decodervae/")]
            # for v in output_vars:
            #     print(type(v.name), v.name)
            # print(len(output_vars))
            # self.train_op = optimizer.minimize(loss_op, var_list=output_vars, global_step=self.global_step)

    def _init_summary(self):
        with tf.variable_scope('summary'):
            tf.summary.scalar("trans_loss", self.loss)
            tf.summary.scalar("merged_mse", self.merged_mse)
            # tf.summary.scalar("trans_loss_mean", self.loss_mean)
            # tf.summary.scalar("trans_loss_logvar", self.loss_logvar)
            # tf.summary.scalar("merged_loss", self.merged_loss)
            tf.summary.scalar("wasserstein_loss", self.wasserstein_loss)
            tf.summary.histogram("z_predition", self.predition)
            self.merged_summary_op = tf.summary.merge_all()
        
    def _gradient_clipping(self, loss_op):
        params = tf.trainable_variables()
        gradients = tf.gradients(loss_op, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, args.clip_norm)
        return clipped_gradients, params

    """
        train_session:
        merged_train_session:

    """

    def train_session(self, sess, feed_dict, loss_op):
        """
            [TODO](xzhren): when nofix encoder & decoder parameters will it be changed ?
        """
        _, summaries, loss, step = sess.run(
            [self.train_op, self.merged_summary_op, loss_op, self.global_step],
                feed_dict)
        return {'summaries': summaries, 'trans_loss': loss, 'step': step}


    # def merged_train_session(self, sess, encoder_model, decoder_model,  x_enc_inp, x_dec_inp, x_dec_out, y_enc_inp, y_dec_inp, y_dec_out):
    #     # merged_summary_op = tf.summary.merge_all()
    #     feed_dict = {
    #         encoder_model.enc_inp: x_enc_inp,
    #         encoder_model.dec_inp: x_dec_inp,
    #         encoder_model.dec_out: x_dec_out,
    #         decoder_model.enc_inp: y_enc_inp,
    #         decoder_model.dec_inp: y_dec_inp,
    #         decoder_model.dec_out: y_dec_out
    #     }
    #     _, summaries, loss, trans_loss, encoder_loss, decoder_loss, step = sess.run(
    #         [self.merged_train_op, self.merged_summary_op, self.merged_loss, self.loss, encoder_model.loss, decoder_model.loss, self.global_step],
    #             feed_dict)
    #     return {'summaries': summaries, 'merged_loss': loss, 'trans_loss': trans_loss, 
    #         'encoder_loss': encoder_loss, 'decoder_loss': decoder_loss, 'step': step}

    def sample_test(self, sess, sentence, answer, encoder_model, decoder_model, predicted_ids_op):
        idx2word = encoder_model.params['idx2word']
        infos = 'I: %s\n' % ' '.join([idx2word[idx] for idx in sentence if idx != PAD_TOKEN])
        # predict_decoder_z = sess.run(self.predition, {encoder_model.enc_inp: np.atleast_2d(sentence)})
        predicted_ids = sess.run(predicted_ids_op, {encoder_model.enc_inp: np.atleast_2d(sentence), decoder_model.enc_seq_len: [args.max_len], decoder_model._batch_size: 1})[0]
        # predicted_ids = sess.run(predicted_ids_op, 
            # {decoder_model._batch_size: 1, decoder_model.z: predict_decoder_z, decoder_model.enc_seq_len: [args.max_len]})[0]
        idx2word = decoder_model.params['idx2word']
        infos += 'O: %s\n' % ' '.join([idx2word[idx] for idx in predicted_ids if idx != PAD_TOKEN])
        infos += 'A: %s\n' % ' '.join([idx2word[idx] for idx in answer if idx != PAD_TOKEN])
        infos += '-'*12 + '\n'
        return infos
