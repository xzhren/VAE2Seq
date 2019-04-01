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
        self.build_trans_loss(params['loss_type'])
        # self.print_parameters()


    def print_parameters(self):
        print("print_parameters:")
        for item in tf.global_variables():
            print('%s: %s' % (item.name, item.get_shape()))
    
    def _build_inputs(self):
        with tf.variable_scope('placeholder'):
            # placeholders x
            self.x_enc_inp = tf.placeholder(tf.int32, [None, args.enc_max_len], name="x_enc_inp")
            self.x_dec_inp = tf.placeholder(tf.int32, [None, args.enc_max_len+1], name="x_dec_inp")
            self.x_dec_out = tf.placeholder(tf.int32, [None, args.enc_max_len+1], name="x_dec_out")
            # placeholders y
            self.y_enc_inp = tf.placeholder(tf.int32, [None, args.dec_max_len], name="y_enc_inp")
            self.y_dec_inp = tf.placeholder(tf.int32, [None, args.dec_max_len+1], name="y_dec_inp")
            self.y_dec_out = tf.placeholder(tf.int32, [None, args.dec_max_len+1], name="y_dec_out")
            # attention data
            self.attention_data = tf.placeholder(tf.int32, [None, args.dec_max_len+1, args.enc_max_len], name="atten_data")
            # train step
            self.global_step = tf.Variable(0, trainable=False)

    def _init_models(self, params):
        # self.global_step = None
        with tf.variable_scope('encodervae'):
            encodervae_inputs = (self.x_enc_inp, self.x_dec_inp, self.x_dec_out, self.global_step)
            params['max_len'] = args.enc_max_len
            params['max_dec_len'] = args.enc_max_len + 1
            self.encoder_model = BaseVAE(params, encodervae_inputs, "encoder")
        with tf.variable_scope('decodervae'):
            decodervae_inputs = (self.y_enc_inp, self.y_dec_inp, self.y_dec_out, self.global_step)
            params['max_len'] = args.dec_max_len
            params['max_dec_len'] = args.dec_max_len + 1
            
            if args.isPointer:
                self.decoder_model = BaseVAE(params, decodervae_inputs, "decoder", 
                            self.encoder_model.encoder_outputs, self.encoder_model.enc_seq_len, self.attention_data)
            elif args.isContext:
                self.decoder_model = BaseVAE(params, decodervae_inputs, "decoder", self.encoder_model.encoder_outputs)
            else:
                self.decoder_model = BaseVAE(params, decodervae_inputs, "decoder")

        with tf.variable_scope('transformer'):
            self.transformer = Transformer(self.encoder_model, self.decoder_model, params['graph_type'], self.global_step)
        with tf.variable_scope('decodervae/decoding', reuse=True):
            self.training_logits = self.decoder_model._decoder_training(self.transformer.predition, reuse=True)
            if args.isPointer:
                self.mask, self.attens_ids, self.predicted_ids = self.decoder_model._decoder_inference(self.transformer.predition)
            else:
                self.predicted_ids_op, _ = self.decoder_model._decoder_inference(self.transformer.predition)
    
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
            mask_fn = lambda l : tf.sequence_mask(l, args.dec_max_len + 1, dtype=tf.float32)
            dec_seq_len = tf.count_nonzero(self.y_dec_out, 1, dtype=tf.int32)
            mask = mask_fn(dec_seq_len) # b x t = 64 x ?
            self.merged_loss_seq =  tf.reduce_sum(tf.contrib.seq2seq.sequence_loss(
                logits = self.training_logits,
                targets = self.y_dec_out,
                weights = mask,
                average_across_timesteps = False,
                average_across_batch = True))
            if model_type == 0:
                self.merged_loss = self.transformer.merged_mse*1000 + self.encoder_model.loss + self.decoder_model.loss
                self.merged_loss_transformer = self.transformer.merged_mse
            elif model_type == 1:
                self.merged_loss = self.transformer.wasserstein_loss*1000 + self.encoder_model.loss + self.decoder_model.loss
                self.merged_loss_transformer = self.transformer.wasserstein_loss
            elif model_type == 2:
                self.merged_loss = self.merged_loss_seq + self.encoder_model.loss + self.decoder_model.loss
                self.merged_loss_transformer = self.merged_loss_seq

        with tf.variable_scope('optimizer'):
            # self.global_step = tf.Variable(0, trainable=False)
            clipped_gradients, params = self._gradient_clipping(self.merged_loss)
            self.merged_train_op = tf.train.AdamOptimizer().apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step)

            clipped_gradients, params = self._gradient_clipping(self.merged_loss_transformer)
            self.merged_train_op_transformer = tf.train.AdamOptimizer().apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step)
            
        with tf.variable_scope('summary'):
            tf.summary.scalar("trans_loss", self.merged_loss_seq)
            tf.summary.scalar("merged_loss", self.merged_loss)
            tf.summary.histogram("z_predition", self.transformer.predition)
            self.merged_summary_op = tf.summary.merge_all()

    def build_trans_loss(self, loss_type):
        if self.params['loss_type'] == 0:
            train_loss = self.transformer.merged_mse
        elif self.params['loss_type'] == 1:
            train_loss = self.transformer.wasserstein_loss
        elif self.params['loss_type'] == 2:
            train_loss = self.merged_loss_seq
        with tf.variable_scope('transformer'):
            self.transformer.build_loss(train_loss)

    def show_parameters(self, sess):
        with open("logs/param_log.txt", "a") as f:
            f.write("==============================\n")
            params = tf.trainable_variables()
            train_params_names = [p.name for p in params]
            params_values = sess.run(train_params_names)
            for name, value in zip(train_params_names, params_values):
                # print(name)
                # print(value)
                f.write(name+"\n"+str(value)+"\n")

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
        if self.params['loss_type'] == 0:
            train_loss = self.transformer.merged_mse
        elif self.params['loss_type'] == 1:
            train_loss = self.transformer.wasserstein_loss
        elif self.params['loss_type'] == 2:
            train_loss = self.merged_loss_seq
        log = self.transformer.train_session(sess, feed_dict, train_loss)
        return log
        
    def merged_train(self, sess, x_enc_inp, x_dec_inp, x_dec_out, y_enc_inp, y_dec_inp, y_dec_out):
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

    def merged_seq_train(self, sess, x_enc_inp, x_dec_inp, x_dec_out, y_enc_inp, y_dec_inp, y_dec_out, atten_data):
        feed_dict = {
            self.x_enc_inp: x_enc_inp,
            self.x_dec_inp: x_dec_inp,
            self.x_dec_out: x_dec_out,
            self.y_enc_inp: y_enc_inp,
            self.y_dec_inp: y_dec_inp,
            self.y_dec_out: y_dec_out,
            self.attention_data: atten_data
        }
        _, summaries, loss, trans_loss, encoder_loss, decoder_loss, step = sess.run(
            [self.merged_train_op, self.merged_summary_op, self.merged_loss, self.merged_loss_seq, self.encoder_model.loss, self.decoder_model.loss, self.global_step],
                feed_dict)
        return {'summaries': summaries, 'merged_loss': loss, 'trans_loss': trans_loss, 
            'encoder_loss': encoder_loss, 'decoder_loss': decoder_loss, 'step': step}

    def merged_transformer_train(self, sess, x_enc_inp, x_dec_inp, x_dec_out, y_enc_inp, y_dec_inp, y_dec_out):
        feed_dict = {
            self.x_enc_inp: x_enc_inp,
            self.x_dec_inp: x_dec_inp,
            self.x_dec_out: x_dec_out,
            self.y_enc_inp: y_enc_inp,
            self.y_dec_inp: y_dec_inp,
            self.y_dec_out: y_dec_out
        }
        _, summaries, loss, step = sess.run(
            [self.merged_train_op_transformer, self.merged_summary_op, self.merged_loss_transformer, self.global_step],
                feed_dict)
        return {'summaries': summaries, 'trans_loss': loss, 'step': step}

    def show_encoder(self, sess, x, y, LOGGER):
        # self.encoder_model.generate(sess)
        infos = self.encoder_model.reconstruct(sess, x, y)
        # self.encoder_model.customized_reconstruct(sess, 'i love this film and i think it is one of the best films')
        # self.encoder_model.customized_reconstruct(sess, 'this movie is a waste of time and there is no point to watch it') 
        LOGGER.write(infos)
        print(infos.strip())

    def show_decoder(self, sess, x, y, LOGGER, x_raw):
        # self.decoder_model.generate(sess)
        feeddict = {}
        feeddict[self.x_enc_inp] = np.atleast_2d(x_raw)
        feeddict[self.y_enc_inp] = np.atleast_2d(x)
        infos = self.decoder_model.reconstruct(sess, x, y, feeddict)
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
        #### method - I
        # batch_size, trans_input = sess.run([self.encoder_model._batch_size, self.encoder_model.z], {self.x_enc_inp:enc_inp})
        # predicted_decoder_z = sess.run(self.transformer.predition, {self.transformer.input:trans_input})
        #### method - I.2.0
        # batch_size, trans_input_mean, trans_input_logvar = sess.run([self.encoder_model._batch_size, self.encoder_model.z_mean, self.encoder_model.z_logvar], {self.x_enc_inp:enc_inp})
        # predicted_decoder_z = sess.run(self.transformer.predition, {self.transformer.input_mean:trans_input_mean, self.transformer.input_logvar:trans_input_logvar})
        # print("========================")
        # print(trans_input)
        # print("------------------------")
        # print(predicted_decoder_z)
        # print("========================")

        # predicted_ids_lt = sess.run(self.predicted_ids_op, 
        #     {self.decoder_model._batch_size: batch_size, self.decoder_model.z: predicted_decoder_z,
        #         self.decoder_model.enc_seq_len: [args.dec_max_len]})

        batch_size = sess.run(self.encoder_model._batch_size, {self.x_enc_inp:enc_inp})
        predicted_ids_lt = sess.run(self.predicted_ids_op, {self.x_enc_inp:enc_inp, self.decoder_model.enc_seq_len: [args.dec_max_len], self.decoder_model._batch_size: batch_size})

        #### method - II
        # batch_size = sess.run(self.encoder_model._batch_size, {self.x_enc_inp:enc_inp})
        # predicted_ids_lt = sess.run(self.predicted_ids_op, 
        #     {self.decoder_model._batch_size: batch_size, self.x_enc_inp: enc_inp, self.y_enc_inp: enc_inp, self.decoder_model.enc_seq_len: [args.dec_max_len]})
        for predicted_ids in predicted_ids_lt:
            with open(outputfile, "a") as f:
                result = ' '.join([idx2word[idx] for idx in predicted_ids])
                end_index = result.find(" </S> ")
                if end_index != -1:
                    result = result[:end_index]
                f.write('%s\n' % result)
                # f.write('%s\n' % ' '.join([idx2word[idx] for idx in predicted_ids]))

    def export_vectors(self, sess, enc_inp, dec_inp):
        code_mean, code_logvar = sess.run(
            [self.transformer.predition_mean, self.transformer.predition_logvar], 
            {self.x_enc_inp:enc_inp})
        desc_mean, desc_logvar = sess.run(
            [self.decoder_model.z_mean, self.decoder_model.z_logvar], 
            {self.y_enc_inp:dec_inp})
        return code_mean, code_logvar, desc_mean, desc_logvar


    def evaluation_pointer(self, sess, enc_inp, outputfile, raw_inp):
        idx2word = self.params['idx2word']
        
        masks, aids, pids = sess.run([self.mask, self.attens_ids, self.predicted_ids], 
            {self.x_enc_inp:enc_inp, self.decoder_model.enc_seq_len: [args.dec_max_len], self.decoder_model._batch_size: args.batch_size})

        masks, aids, pids = masks[:,:,0], aids[:,:,0], pids[:,:,0]
        for i, (mask, aid, pid) in enumerate(zip(masks, aids, pids)):
            # print(i, mask, aid, pid)
            # print(raw_inp[i])
            with open(outputfile, "a") as f:
                result = ''
                for m, a, p in zip(mask, aid, pid):
                    # print(m, a, p)
                    if m == 1: result += idx2word[p] + " "
                    elif m ==0:
                        if a >= len(raw_inp[i]):
                            result += " UNK "
                        else:
                            result += raw_inp[i][a] + " "
                    else: print("ERRRRRRRRRRORR!!!")
                # result = ' '.join([idx2word[idx] for idx in predicted_ids])
                result = result.strip()
                # print(result)
                end_index = result.find(" </S> ")
                if end_index != -1:
                    result = result[:end_index]
                f.write('%s\n' % result)
                # f.write('%s\n' % ' '.join([idx2word[idx] for idx in predicted_ids]))