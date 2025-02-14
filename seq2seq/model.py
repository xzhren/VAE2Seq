from config import args
from modified import ModifiedBasicDecoder, ModifiedBeamSearchDecoder

import tensorflow as tf
import numpy as np
import sys

from data_reddit import START_TOKEN, END_TOKEN, UNK_STRING, PAD_STRING

class VRAE:
    def __init__(self, params):
        self.params = params
        self._build_graph()
        self._init_summary()

    def _build_graph(self):
        self._build_forward_graph()
        self._build_backward_graph()

    def _build_forward_graph(self):
        self._build_inputs()
        encoded_state = self._encode()
        self._decode(encoded_state)

    def _build_backward_graph(self):
        self.global_step = tf.Variable(0, trainable=False)

        self.nll_loss = self._nll_loss_fn()
        self.loss = self.nll_loss
        
        clipped_gradients, params = self._gradient_clipping(self.loss)
        self.train_op = tf.train.AdamOptimizer().apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step)

    def _init_summary(self):
        tf.summary.scalar("loss", self.loss)
        self.merged_summary_op = tf.summary.merge_all()

    def _build_inputs(self):
        # placeholders
        self.enc_inp = tf.placeholder(tf.int32, [None, args.max_len])
        self.dec_inp = tf.placeholder(tf.int32, [None, args.max_dec_len+1])
        self.dec_out = tf.placeholder(tf.int32, [None, args.max_dec_len+1])
        # global helpers
        self._batch_size = tf.shape(self.enc_inp)[0]
        self.enc_seq_len = tf.count_nonzero(self.enc_inp, 1, dtype=tf.int32)
        self.dec_seq_len = tf.count_nonzero(self.dec_out, 1, dtype=tf.int32)
    
    def _encode(self):
        # the embedding is shared between encoder and decoder
        # since the source and the target for an autoencoder are the same
        with tf.variable_scope('encoder'):
            if args.diff_input:
                tied_embedding = tf.get_variable('tied_embedding_encoder',
                    [self.params['vocab_size_encoder'], args.embedding_dim],
                    tf.float32)
            else:
                tied_embedding = tf.get_variable('tied_embedding',
                    [self.params['vocab_size'], args.embedding_dim],
                    tf.float32)
            _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = self._rnn_cell(args.rnn_size//2),
                cell_bw = self._rnn_cell(args.rnn_size//2), 
                inputs = tf.nn.embedding_lookup(tied_embedding, self.enc_inp),
                sequence_length = self.enc_seq_len,
                dtype = tf.float32)
        encoded_state = tf.concat((state_fw, state_bw), -1)
        return encoded_state

    def _decode(self, encoded_state):
        self.training_rnn_out, self.training_logits = self._decoder_training(encoded_state)
        self.predicted_ids = self._decoder_inference(encoded_state)

    def _dynamic_time_pad(self, tensor, max_length):
        shape = tensor.get_shape().as_list()
        shape_op = tf.shape(tensor)

        pad_size = max_length - shape_op[1]

        if len(shape) == 3:
            tensor = tf.concat([
                tensor,
                tf.zeros([shape_op[0], pad_size, shape[2]], dtype=tensor.dtype)
            ], 1)
        elif len(shape) == 2:
            tensor = tf.concat([
                tensor,
                tf.zeros([shape_op[0], pad_size], dtype=tensor.dtype)
            ], 1)
        else:
            raise NotImplemented(f'tensor with {len(shape)} dimentions.')

        return tensor

    def _decoder_training(self, encoded_state):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            tied_embedding = tf.get_variable('tied_embedding', [self.params['vocab_size'], args.embedding_dim])

        with tf.variable_scope('decoding'):
            init_state = tf.layers.dense(encoded_state, args.rnn_size, tf.nn.elu)

            # lin_proj = tf.layers.Dense(self.params['vocab_size'], _scope='decoder/dense')
            
            helper = tf.contrib.seq2seq.TrainingHelper(
                inputs = tf.nn.embedding_lookup(tied_embedding, self.dec_inp),
                sequence_length = self.dec_seq_len)
            print("helper:", helper)
            decoder = ModifiedBasicDecoder(
                cell = self._rnn_cell(),
                helper = helper,
                initial_state = init_state,
                concat_z = encoded_state)
            print("decoder:", decoder)
            # b x t x h
            decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = decoder)
            print("decoder_output:", decoder_output) # 
            logits = self._dynamic_time_pad(decoder_output.rnn_output, args.max_dec_len+1)
            print("logits:", logits) # 
        
            with tf.variable_scope('decoder'):
                # b x t x h => b x t x v ? # 64,?,200
                print("decoder_output.rnn_output:", logits)
                lin_proj = tf.layers.dense(logits, self.params['vocab_size'], name="dense")

        # return decoder_output.rnn_output, lin_proj.apply(decoder_output.rnn_output)
        return logits, lin_proj


    def _decoder_inference(self, encoded_state):
        tiled_z = tf.tile(tf.expand_dims(encoded_state, 1), [1, args.beam_width, 1])
        
        with tf.variable_scope('encoder', reuse=True):
            tied_embedding = tf.get_variable('tied_embedding', [self.params['vocab_size'], args.embedding_dim])

        with tf.variable_scope('decoding', reuse=True):
            init_state = tf.layers.dense(encoded_state, args.rnn_size, tf.nn.elu, reuse=True)

            decoder = ModifiedBeamSearchDecoder(
                cell = self._rnn_cell(reuse=True),
                embedding = tied_embedding,
                start_tokens = tf.tile(tf.constant(
                    [self.params['word2idx']['<S>']], dtype=tf.int32), [self._batch_size]),
                end_token = self.params['word2idx']['</S>'],
                initial_state = tf.contrib.seq2seq.tile_batch(init_state, args.beam_width),
                beam_width = args.beam_width,
                output_layer = tf.layers.Dense(self.params['vocab_size'], _reuse=True),
                concat_z = tiled_z)
            decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = decoder,
                maximum_iterations = 2*tf.reduce_max(self.enc_seq_len))

        return decoder_output.predicted_ids[:, :, 0]


    def _rnn_cell(self, rnn_size=None, reuse=False):
        rnn_size = args.rnn_size if rnn_size is None else rnn_size
        return tf.nn.rnn_cell.GRUCell(
            rnn_size, kernel_initializer=tf.orthogonal_initializer(), reuse=reuse)


    def _nll_loss_fn(self):
        # mask_fn = lambda l : tf.sequence_mask(l, tf.reduce_max(l), dtype=tf.float32)
        mask_fn = lambda l : tf.sequence_mask(l, args.max_dec_len+1, dtype=tf.float32)
        mask = mask_fn(self.dec_seq_len) # b x t 64 x ?
        if (args.num_sampled <= 0) or (args.num_sampled >= self.params['vocab_size']):
            return tf.reduce_sum(tf.contrib.seq2seq.sequence_loss(
                logits = self.training_logits,
                targets = self.dec_out,
                weights = mask,
                average_across_timesteps = False,
                average_across_batch = True))
        else:
            with tf.variable_scope('decoding/decoder/dense', reuse=True):
                print("tf.get_variable('kernel'):", tf.get_variable('kernel')) # 200 x 35006
                print("tf.get_variable('bias'):", tf.get_variable('bias')) # 35006
                print("mask:", mask)
                mask = tf.reshape(mask, [-1])
                print("mask:", mask) # 3200 - 64 x 151 = 9664
                print("tf.reshape(self.dec_out, [-1, 1]):", tf.reshape(self.dec_out, [-1, 1])) # 9664 x 1
                print("tf.reshape(self.training_rnn_out, [-1, args.rnn_size]):", tf.reshape(self.training_rnn_out, [-1, args.rnn_size])) # ? x 200
                print("args.num_sampled:", args.num_sampled)
                return tf.reduce_sum(mask * tf.nn.sampled_softmax_loss(
                    weights = tf.transpose(tf.get_variable('kernel')),
                    biases = tf.get_variable('bias'),
                    labels = tf.reshape(self.dec_out, [-1, 1]),
                    inputs = tf.reshape(self.training_rnn_out, [-1, args.rnn_size]),
                    num_sampled = args.num_sampled,
                    num_classes = self.params['vocab_size'],
                )) / tf.to_float(self._batch_size)


    def train_session(self, sess, enc_inp, dec_inp, dec_out):
        _, summaries, loss, step = sess.run(
            [self.train_op, self.merged_summary_op, self.loss, self.global_step],
                {self.enc_inp: enc_inp, self.dec_inp: dec_inp, self.dec_out: dec_out})
        return {'summaries': summaries, 'loss': loss, 'step': step}


    def evaluation(self, sess, enc_inp, outputfile):
        idx2word = self.params['idx2word']
        predicted_ids_lt = sess.run(self.predicted_ids, {self.enc_inp:enc_inp})
        for predicted_ids in predicted_ids_lt:
            with open(outputfile, "a") as f:
                result = ' '.join([idx2word[idx] for idx in predicted_ids])
                end_index = result.find(" </S> ")
                if end_index != -1:
                    result = result[:end_index]
                f.write('%s\n' % result)

    def reconstruct(self, sess, sentence, reference):
        idx2word = self.params['idx2word']
        if args.diff_input:
            print('I: %s' % ' '.join([self.params['idx2token'][idx] for idx in sentence]))
        else:
            print('I: %s' % ' '.join([idx2word[idx] for idx in sentence]))
        print()
        print('R: %s' % ' '.join([idx2word[idx] for idx in reference]))
        print()
        predicted_ids = sess.run(self.predicted_ids, {self.enc_inp: np.atleast_2d(sentence)})[0]
        print('O: %s' % ' '.join([idx2word[idx] for idx in predicted_ids]))
        print('-'*12)

    def _gradient_clipping(self, loss_op):
        params = tf.trainable_variables()
        gradients = tf.gradients(loss_op, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, args.clip_norm)
        return clipped_gradients, params