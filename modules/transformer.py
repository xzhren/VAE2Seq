import tensorflow as tf
import numpy as np

from config import args
from data.data_reddit import PAD_TOKEN
from modules.wasserstein_utils import wasserstein_loss

class Transformer:
    def __init__(self, encoder, decoder):
        self.input_mean = encoder.z_mean
        self.output_mean = decoder.z_mean
        self.input_logvar = encoder.z_logvar
        self.output_logvar = decoder.z_logvar
        self.input = encoder.z
        self.output = decoder.z

        self.encoder_class_num = 300
        self.decoder_class_num = 300

        # self._build_graph(encoder.loss, decoder.loss)
        self._build_graph_mlp(encoder.loss, decoder.loss)
        self._init_summary()

    def _build_graph(self, encoder_loss, decoder_loss):
        # input: b x l
        # [o]: predition: b x l
        with tf.variable_scope('trans_encoder'):
            trans_encoder_embedding = tf.get_variable('trans_encoder_embedding',
                [self.encoder_class_num, args.latent_size],
                tf.float32)
            input_dist_mean = tf.nn.softmax(tf.matmul(self.input_mean, trans_encoder_embedding, transpose_b=True)) # b x encoder_class_num
            input_dist_logvar = tf.nn.softmax(tf.matmul(self.input_logvar, trans_encoder_embedding, transpose_b=True)) # b x encoder_class_num
        with tf.variable_scope('trans'):
            trans_w = tf.get_variable('trans_w', [self.encoder_class_num, self.decoder_class_num], tf.float32)
            input_dist_mean = tf.nn.softmax(tf.matmul(input_dist_mean, trans_w)) # b x decoder_class_num
            input_dist_logvar = tf.nn.softmax(tf.matmul(input_dist_logvar, trans_w)) # b x decoder_class_num
        with tf.variable_scope('trans_decoder'):
            trans_decoder_embedding = tf.get_variable('trans_decoder_embedding',
                [self.decoder_class_num, args.latent_size],
                tf.float32)
            self.predition_mean = tf.matmul(input_dist_mean, trans_decoder_embedding) # b x l
            self.predition_logvar = tf.matmul(input_dist_logvar, trans_decoder_embedding) # b x l
            self.predition = self.predition_mean + tf.exp(0.5 * self.predition_logvar) * tf.truncated_normal(tf.shape(self.predition_logvar))
        # print("trans, input:", self.input)
        # print("trans, predition_mean:", self.predition_mean)
        # print("trans, predition_logvar:", self.predition_logvar)
        # print("trans, predition:", self.predition)
        # with tf.variable_scope('mlp'):
        #     self.predition = tf.layers.dense(self.input, args.latent_size)
        with tf.variable_scope('loss'):
            self.loss_mean = tf.losses.mean_squared_error(self.predition_mean, self.output_mean)
            self.loss_logvar = tf.losses.mean_squared_error(self.predition_logvar, self.output_logvar)
            self.loss = tf.losses.mean_squared_error(self.predition, self.output)
            self.merged_loss = (self.loss_mean+self.loss_logvar+self.loss)*1000 + encoder_loss + decoder_loss
            self.wasserstein_loss = wasserstein_loss(self.predition, self.output)
        
        # with tf.variable_scope('optimizer'):
        #     self.global_step = tf.Variable(0, trainable=False)
        #     clipped_gradients, params = self._gradient_clipping(self.loss)
        #     self.train_op = tf.train.AdamOptimizer().apply_gradients(
        #         zip(clipped_gradients, params), global_step=self.global_step)

        #     clipped_gradients, params = self._gradient_clipping(self.merged_loss)
        #     self.merged_train_op = tf.train.AdamOptimizer().apply_gradients(
        #         zip(clipped_gradients, params), global_step=self.global_step)
    
    def _build_graph_mlp(self, encoder_loss, decoder_loss):
        # input: b x l
        # [o]: predition: b x l
        with tf.variable_scope('trans_mlp'):
            trans_mean = tf.layers.dense(self.input_mean, args.latent_size, tf.nn.tanh, name="tran_mean_1")
            trans_mean = tf.layers.dense(trans_mean, args.latent_size, tf.nn.tanh, name="tran_mean_2")
            trans_mean = tf.layers.dense(trans_mean, args.latent_size, tf.nn.tanh, name="tran_mean_3")

            trans_logvar = tf.layers.dense(self.input_logvar, args.latent_size, tf.nn.tanh, name="trans_logvar_1")
            trans_logvar = tf.layers.dense(trans_logvar, args.latent_size, tf.nn.tanh, name="tran_logvar_2")
            trans_logvar = tf.layers.dense(trans_logvar, args.latent_size, tf.nn.tanh, name="tran_logvar_3")

            self.predition_mean = tf.layers.dense(trans_mean, args.latent_size, tf.nn.tanh, name="tran_mean") + 1e-12 # b x l
            self.predition_logvar = tf.layers.dense(trans_logvar, args.latent_size, tf.nn.tanh, name="tran_logvar") + 1e-12 # b x l
            self.predition = self.predition_mean + tf.exp(0.5 * self.predition_logvar) * tf.truncated_normal(tf.shape(self.predition_logvar))


        with tf.variable_scope('loss'):
            self.loss_mean = tf.losses.mean_squared_error(self.predition_mean, self.output_mean)
            self.loss_logvar = tf.losses.mean_squared_error(self.predition_logvar, self.output_logvar)
            self.loss = tf.losses.mean_squared_error(self.predition, self.output)
            self.merged_loss = (self.loss_mean+self.loss_logvar+self.loss)*1000 + encoder_loss + decoder_loss
            self.merged_mse = (self.loss_mean+self.loss_logvar+self.loss)
            self.wasserstein_loss = wasserstein_loss(self.predition, self.output)    

    def _init_summary(self):
        with tf.variable_scope('summary'):
            tf.summary.scalar("trans_loss", self.loss)
            tf.summary.scalar("trans_loss_mean", self.loss_mean)
            tf.summary.scalar("trans_loss_logvar", self.loss_logvar)
            tf.summary.scalar("merged_loss", self.merged_loss)
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

    def train_session(self, sess, encoder_model, decoder_model,  x_enc_inp, x_dec_inp, x_dec_out, y_enc_inp, y_dec_inp, y_dec_out):
        """
            [TODO](xzhren): when nofix encoder & decoder parameters will it be changed ?
        """
        feed_dict = {
            encoder_model.enc_inp: x_enc_inp,
            encoder_model.dec_inp: x_dec_inp,
            encoder_model.dec_out: x_dec_out,
            decoder_model.enc_inp: y_enc_inp,
            decoder_model.dec_inp: y_dec_inp,
            decoder_model.dec_out: y_dec_out
        }
        _, summaries, loss, step = sess.run(
            [self.train_op, self.merged_summary_op, self.loss, self.global_step],
                feed_dict)
        return {'summaries': summaries, 'trans_loss': loss, 'step': step}


    def merged_train_session(self, sess, encoder_model, decoder_model,  x_enc_inp, x_dec_inp, x_dec_out, y_enc_inp, y_dec_inp, y_dec_out):
        # merged_summary_op = tf.summary.merge_all()
        feed_dict = {
            encoder_model.enc_inp: x_enc_inp,
            encoder_model.dec_inp: x_dec_inp,
            encoder_model.dec_out: x_dec_out,
            decoder_model.enc_inp: y_enc_inp,
            decoder_model.dec_inp: y_dec_inp,
            decoder_model.dec_out: y_dec_out
        }
        _, summaries, loss, trans_loss, encoder_loss, decoder_loss, step = sess.run(
            [self.merged_train_op, self.merged_summary_op, self.merged_loss, self.loss, encoder_model.loss, decoder_model.loss, self.global_step],
                feed_dict)
        return {'summaries': summaries, 'merged_loss': loss, 'trans_loss': trans_loss, 
            'encoder_loss': encoder_loss, 'decoder_loss': decoder_loss, 'step': step}

    def sample_test(self, sess, sentence, answer, encoder_model, decoder_model, predicted_ids_op):
        idx2word = encoder_model.params['idx2word']
        infos = 'I: %s\n' % ' '.join([idx2word[idx] for idx in sentence if idx != PAD_TOKEN])
        predict_decoder_z = sess.run(self.predition, {encoder_model.enc_inp: np.atleast_2d(sentence)})
        predicted_ids = sess.run(predicted_ids_op, 
            {decoder_model._batch_size: 1, decoder_model.z: predict_decoder_z, decoder_model.enc_seq_len: [args.max_len]})[0]
        idx2word = decoder_model.params['idx2word']
        infos += 'O: %s\n' % ' '.join([idx2word[idx] for idx in predicted_ids if idx != PAD_TOKEN])
        infos += 'A: %s\n' % ' '.join([idx2word[idx] for idx in answer if idx != PAD_TOKEN])
        infos += '-'*12 + '\n'
        return infos
