import tensorflow as tf
import numpy as np

from config import args
from modules.modified import ModifiedBasicDecoder, ModifiedBeamSearchDecoder
from modules.modified import ContextDecoder, ContextBeamSearchDecoder
from modules.modified import PointerDecoder, PointerBeamSearchDecoder
from data.data_reddit import START_TOKEN, END_TOKEN, PAD_TOKEN, UNK_STRING, PAD_STRING

class BaseVAE:
    def __init__(self, params, inputs, prefix, 
            context_encoder_ouputs=None, 
            x_enc_inp_oovs=None, max_oovs=None):
        self.prefix = prefix
        self.params = params
        self.context_encoder_ouputs = context_encoder_ouputs
        # self.enc_atten_len = enc_atten_len
        # self.enc_atten_label = enc_atten_label
        self.x_enc_inp_oovs = x_enc_inp_oovs
        self.max_oovs = max_oovs
        if prefix == "decoder":
            self.isContext = args.isContext
            self.isPointer = args.isPointer
        else:
            self.isContext = False
            self.isPointer = False

        if self.isContext:
            assert self.isPointer == False
            assert self.context_encoder_ouputs != None
        if self.isPointer:
            assert self.isContext == False
            assert self.context_encoder_ouputs != None
            assert self.x_enc_inp_oovs != None
            assert self.max_oovs != None
        self._build_inputs(inputs)
        self._build_graph(prefix)
        self._init_summary(prefix)

    def _build_inputs(self, inputs):
        with tf.variable_scope('placeholder'):
            # placeholders
            self.enc_inp = inputs[0]
            self.dec_inp = inputs[1]
            self.dec_out = inputs[2]
            self.global_step = inputs[3]
            # global helpers
            self._batch_size = tf.shape(self.enc_inp)[0]
            self.enc_seq_len = tf.count_nonzero(self.enc_inp, 1, dtype=tf.int32)
            self.dec_seq_len = tf.count_nonzero(self.dec_out, 1, dtype=tf.int32)

    def _build_graph(self, prefix):
        encoded_state = self._encode()
        z = self._reparam(encoded_state)
        self._decode(z)

        with tf.variable_scope('loss'):
            # self.global_step = tf.Variable(0, trainable=False)
            self.nll_loss = self._nll_loss_fn()
            # if self.isPointer:
            #     self.nll_loss += self._nll_loss_fn_pointer()
            #     self.pointer_loss = self._nll_loss_fn_pointer()
            #     self.raw_nll_loss = self._nll_loss_fn()

            self.kl_w = self._kl_w_fn(args.anneal_max, args.anneal_bias, self.global_step)
            self.kl_loss = self._kl_loss_fn(self.z_mean, self.z_logvar)
        
            #######
            loss_op = self.nll_loss + self.kl_w * self.kl_loss
            # loss_op = self.nll_loss
            ######
            self.loss = loss_op
        
        with tf.variable_scope('optimizer'):
            clipped_gradients, params = self._gradient_clipping(loss_op)
            print("======== [base vae params]", prefix, len(clipped_gradients))
            if prefix == "decoder":
                clipped_gradients_, params_ = [], []
                for k, v in zip(clipped_gradients, params):
                    if v.name.startswith("decodervae/"):
                        clipped_gradients_.append(k)
                        params_.append(v)
                clipped_gradients, params = clipped_gradients_, params_
            elif prefix == "encoder":
                clipped_gradients_, params_ = [], []
                for k, v in zip(clipped_gradients, params):
                    if v.name.startswith("encodervae/"):
                        clipped_gradients_.append(k)
                        params_.append(v)
                clipped_gradients, params = clipped_gradients_, params_
            for k, v in zip(clipped_gradients, params):
                print(v.name)
            print("======== [end]", len(clipped_gradients))
            self.train_op = tf.train.AdamOptimizer().apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step)

    def _init_summary(self, prefix):
        with tf.variable_scope('summary'):
            tf.summary.scalar(prefix+"_nll_loss", self.nll_loss)
            tf.summary.scalar(prefix+"_kl_w", self.kl_w)
            tf.summary.scalar(prefix+"_kl_loss", self.kl_loss)
            tf.summary.scalar(prefix+"_loss", self.loss)
            tf.summary.histogram(prefix+"_z_mean", self.z_mean)
            tf.summary.histogram(prefix+"_z_logvar", self.z_logvar)
            tf.summary.histogram(prefix+"_z", self.z)
            if self.isPointer:
                # tf.summary.scalar(prefix+"_pointer_loss", self.pointer_loss)
                # tf.summary.scalar(prefix+"_raw_nll_loss", self.raw_nll_loss)
                tf.summary.image(prefix+"attention", tf.expand_dims(self.attens, -1))
                # tf.summary.image("attention", tf.expand_dims(attention[:1], -1))
            self.merged_summary_op = tf.summary.merge_all()

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
            bi_encoder_outputs, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = self._rnn_cell(args.rnn_size//2),
                cell_bw = self._rnn_cell(args.rnn_size//2), 
                inputs = tf.nn.embedding_lookup(tied_embedding, self.enc_inp),
                sequence_length = self.enc_seq_len,
                dtype = tf.float32)
            self.encoder_outputs = tf.concat(bi_encoder_outputs, axis=2)

            encoded_state = tf.concat((state_fw, state_bw), -1)
            if args.diff_input:
                self.tied_embedding = tf.get_variable('tied_embedding',
                    [self.params['vocab_size'], args.embedding_dim],
                    tf.float32)
            else:
                self.tied_embedding = tied_embedding
        return encoded_state

    def _reparam(self, encoded_state):
        with tf.variable_scope('reparam'):
            self.z_mean = tf.layers.dense(encoded_state, args.latent_size)
            self.z_logvar = tf.layers.dense(encoded_state, args.latent_size)
            
            self.gaussian_noise = tf.truncated_normal(tf.shape(self.z_logvar))
            self.z = self.z_mean + tf.exp(0.5 * self.z_logvar) * self.gaussian_noise
        return self.z

    def _decode(self, z):
        with tf.variable_scope('decoding'):
            self.training_logits = self._decoder_training(z)
            # if self.isPointer:
            #     self._decoder_inference(z)
            # else:
            self.predicted_ids, _ = self._decoder_inference(z)

    def _rnn_cell(self, rnn_size=None, reuse=False):
        rnn_size = args.rnn_size if rnn_size is None else rnn_size
        return tf.nn.rnn_cell.GRUCell(
            rnn_size, kernel_initializer=tf.orthogonal_initializer(), reuse=reuse)

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
            raise Exception(f'tensor with {len(shape)} dimentions.')
        return tensor

    def _decoder_training(self, z, reuse=False):
        tied_embedding = self.tied_embedding
        init_state = tf.layers.dense(z, args.rnn_size, tf.nn.elu, name="z_proj_state", reuse=reuse)
        
        self.decoder_cells = self._rnn_cell(reuse=reuse)

        helper = tf.contrib.seq2seq.TrainingHelper(
            inputs = tf.nn.embedding_lookup(tied_embedding, self.dec_inp),
            sequence_length = self.dec_seq_len)
        if self.isPointer:
            decoder = PointerDecoder(
                cell = self.decoder_cells,
                helper = helper,
                initial_state = init_state,
                concat_z = z,
                encoder_ouputs = self.context_encoder_ouputs)
        elif self.isContext:
            decoder = ContextDecoder(
                cell = self.decoder_cells,
                helper = helper,
                initial_state = init_state,
                concat_z = z,
                encoder_ouputs = self.context_encoder_ouputs)
        else:
            decoder = ModifiedBasicDecoder(
                cell = self.decoder_cells,
                helper = helper,
                initial_state = init_state,
                concat_z = z)
        # b x t x h
        decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder = decoder)
        print("train decoder_output:", decoder_output)
        
        logits = self._dynamic_time_pad(decoder_output.rnn_output, self.params['max_dec_len'])
        # logits = decoder_output.rnn_output
        if self.isPointer:
            logits, attens = tf.split(logits, [args.rnn_size+args.rnn_size, args.enc_max_len], axis=2)
            _, states = tf.split(logits, [args.rnn_size, args.rnn_size], axis=2)
            pointer_proj = tf.layers.Dense(1, activation=tf.sigmoid, _scope='pointer_proj_dense', _reuse=reuse)
            pointer = pointer_proj.apply(states) 
            # print("logits:", logits)
            # print("attens:", attens)
            # print("pointer:", pointer)
        
        # b x t x h => b x t x v ? # 64,?,200
        lin_proj = tf.layers.Dense(self.params['vocab_size'], _scope='out_proj_dense', _reuse=reuse)
        logits_dist = lin_proj.apply(logits) 

        if self.isPointer:
            logits_dist = logits_dist * (1-pointer)    
            self.attens = attens * pointer
            logits_dist = self._calc_final_dist(logits_dist, self.attens)


        return logits_dist

    def _calc_final_dist(self, vocab_dists, attn_dists):
        extra_zeros = tf.zeros((self._batch_size, args.dec_max_len+1, self.max_oovs))
        vocab_dists_extended = tf.concat(axis=2, values=[vocab_dists, extra_zeros])

        extended_vsize = len(self.params['word2idx']) + self.max_oovs
        batch_nums = tf.range(0, limit=self._batch_size) 
        batch_nums = tf.tile(tf.expand_dims(batch_nums, 1), [1, args.dec_max_len+1]) # shape (batch_size, attn_len)
        batch_nums = tf.reshape(batch_nums, [-1])
        # batch_nums = tf.expand_dims(batch_nums, 1) # shape (batch_size, 1)
        # attn_len = tf.shape(self._enc_batch_extend_vocab)[1] # number of states we attend over
        batch_nums = tf.tile(tf.expand_dims(batch_nums, 1), [1, args.enc_max_len]) # shape (batch_size, attn_len)
        x_enc_inp_oovs = tf.tile(tf.expand_dims(self.x_enc_inp_oovs, 1), [1, args.dec_max_len+1, 1]) # shape (batch_size, attn_len)
        x_enc_inp_oovs = tf.reshape(batch_nums, [-1, args.enc_max_len])
        indices = tf.stack( (batch_nums, x_enc_inp_oovs), axis=2) # shape (batch_size, enc_t, 2)
        shape = [self._batch_size*(args.dec_max_len+1), extended_vsize]
        attn_dists =  tf.reshape(attn_dists, [-1, args.enc_max_len])
        attn_dists_projected = tf.scatter_nd(indices, attn_dists, shape)
        print("attn_dists_projected:", attn_dists_projected)
        attn_dists_projected = tf.reshape(attn_dists_projected, [self._batch_size, args.dec_max_len+1, extended_vsize])
        # [32,400,2], [32,101,400], [32,101,50031]

        return vocab_dists_extended + attn_dists_projected

    def _calc_final_dist_decoder(self):
        extra_zeros = tf.zeros((self._batch_size, args.beam_width, self.max_oovs))

        extended_vsize = len(self.params['word2idx']) + self.max_oovs
        batch_nums = tf.range(0, limit=self._batch_size) 
        batch_nums = tf.expand_dims(batch_nums, 1) # shape (batch_size, 1)
        # attn_len = tf.shape(self._enc_batch_extend_vocab)[1] # number of states we attend over
        batch_nums = tf.tile(batch_nums, [1, args.enc_max_len]) # shape (batch_size, attn_len)
        extra_indices = tf.zeros((self._batch_size, args.enc_max_len), dtype=tf.int32) # shape (batch_size, attn_len)
        indices = tf.stack( (batch_nums, extra_indices, self.x_enc_inp_oovs), axis=2) # shape (batch_size, enc_t, 3)
        indices = tf.expand_dims(indices, 1) # shape (batch_size, 1, enc_t, 3)
        indices = tf.tile(indices, [1, args.beam_width, 1, 1]) # shape (batch_size, beam, enc_t, 3)
        shape = [self._batch_size, args.beam_width, extended_vsize]
           
        return extra_zeros, indices, shape

    def _decoder_inference(self, z):
        tied_embedding = self.tied_embedding
        init_state = tf.layers.dense(z, args.rnn_size, tf.nn.elu, name="z_proj_state", reuse=True)
        tiled_z = tf.tile(tf.expand_dims(z, 1), [1, args.beam_width, 1])

        if self.isPointer:
            decoder = PointerBeamSearchDecoder(
                cell = self.decoder_cells,
                embedding = tied_embedding,
                start_tokens = tf.tile(tf.constant(
                    [self.params['word2idx']['<S>']], dtype=tf.int32), [self._batch_size]),
                end_token = self.params['word2idx']['</S>'],
                initial_state = tf.contrib.seq2seq.tile_batch(init_state, args.beam_width),
                beam_width = args.beam_width,
                output_layer = tf.layers.Dense(self.params['vocab_size'], _scope="out_proj_dense", _reuse=True),
                pointer_layer = tf.layers.Dense(1, activation=tf.sigmoid, _scope='pointer_proj_dense', _reuse=True),
                pointer_data = self._calc_final_dist_decoder(),
                concat_z = tiled_z,
                encoder_ouputs = self.context_encoder_ouputs)
        elif self.isContext:
            decoder = ContextBeamSearchDecoder(
                cell = self.decoder_cells,
                embedding = tied_embedding,
                start_tokens = tf.tile(tf.constant(
                    [self.params['word2idx']['<S>']], dtype=tf.int32), [self._batch_size]),
                end_token = self.params['word2idx']['</S>'],
                initial_state = tf.contrib.seq2seq.tile_batch(init_state, args.beam_width),
                beam_width = args.beam_width,
                output_layer = tf.layers.Dense(self.params['vocab_size'], _scope="out_proj_dense", _reuse=True),
                concat_z = tiled_z,
                encoder_ouputs = self.context_encoder_ouputs)
        else:
            decoder = ModifiedBeamSearchDecoder(
                cell = self.decoder_cells,
                embedding = tied_embedding,
                start_tokens = tf.tile(tf.constant(
                    [self.params['word2idx']['<S>']], dtype=tf.int32), [self._batch_size]),
                end_token = self.params['word2idx']['</S>'],
                initial_state = tf.contrib.seq2seq.tile_batch(init_state, args.beam_width),
                beam_width = args.beam_width,
                output_layer = tf.layers.Dense(self.params['vocab_size'], _scope="out_proj_dense", _reuse=True),
                concat_z = tiled_z)
            
        decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder = decoder,
            maximum_iterations = 2*tf.reduce_max(self.enc_seq_len))
        print("inference decoder_output:", decoder_output)
        # print("inference attens:", decoder_output.beam_search_decoder_output.attens) # ?, ?, 5, 400
        # print("inference scores:", decoder_output.beam_search_decoder_output.scores) # ?, ?, 5
    
        return decoder_output.predicted_ids[:, :, 0], decoder_output.beam_search_decoder_output.scores[:, :, 0]    
    
    def _gradient_clipping(self, loss_op):
        params = tf.trainable_variables()
        gradients = tf.gradients(loss_op, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, args.clip_norm)
        return clipped_gradients, params

    def _nll_loss_fn(self):
        # mask_fn = lambda l : tf.sequence_mask(l, tf.reduce_max(l), dtype=tf.float32)
        mask_fn = lambda l : tf.sequence_mask(l, self.params['max_dec_len'], dtype=tf.float32)
        mask = mask_fn(self.dec_seq_len) # b x t = 64 x ?
        return tf.reduce_sum(tf.contrib.seq2seq.sequence_loss(
            logits = self.training_logits,
            targets = self.dec_out,
            weights = mask,
            average_across_timesteps = False,
            average_across_batch = True))

    # def _nll_loss_fn_pointer(self):
    #     mask_fn = lambda l : tf.sequence_mask(l, args.enc_max_len, dtype=tf.float32)
    #     enc_mask = mask_fn(self.enc_atten_len) # b x t = 64 x 400
    #     enc_mask = tf.expand_dims(enc_mask, 1) # b x 1 x t
    #     mask_fn = lambda l : tf.sequence_mask(l, self.params['max_dec_len'], dtype=tf.float32)
    #     dec_mask = mask_fn(self.dec_seq_len) # b x t = 64 x ?
    #     # dec_mask = tf.expand_dims(dec_mask, 2) # b x 1 x t

    #     label = tf.cast(self.enc_atten_label, tf.float32)
    #     attens = self._dynamic_time_pad(self.attens, self.params['max_dec_len'])
    #     # b x dec_l x enc_l
    #     # return tf.reduce_sum(-tf.log(
    #     #    tf.reduce_sum(attens * label * enc_mask, axis=2)+1e-13) * dec_mask)
    #     return tf.reduce_sum(-
    #     tf.reduce_sum(label * tf.log(attens+1e-13) + (1-label) * tf.log(1-attens+1e-13) * enc_mask, axis=2)
    #        * dec_mask)

    def _kl_w_fn(self, anneal_max, anneal_bias, global_step):
        '''
         when anneal_step = 0, kl_w = 0; when anneal_step = anneal_bias, kl_w = 1
        '''
        train_data_len = args.data_len
        epochstep = train_data_len / self._batch_size 
        anneal_step_w = tf.cast(1 / epochstep * anneal_bias, tf.float32)
        return anneal_max * tf.sigmoid((10 / anneal_bias) * (
            anneal_step_w * tf.cast(global_step, tf.float32) - tf.constant(anneal_bias / 2)))

    def _kl_loss_fn(self, mean, gamma):
        return 0.5 * tf.reduce_sum(
            tf.exp(gamma) + tf.square(mean) - 1 - gamma) / tf.to_float(self._batch_size)

    """
        Main Functions
            train_session:
            reconstruct:
            customized_reconstruct:
            evaluation:
            generate:
    """
    
    def train_session(self, sess, feed_dict):
        _, summaries, loss, nll_loss, kl_w, kl_loss, step = sess.run(
            [self.train_op, self.merged_summary_op, self.loss, self.nll_loss, self.kl_w, self.kl_loss, self.global_step],
            feed_dict)
                # {self.enc_inp: enc_inp, self.dec_inp: dec_inp, self.dec_out: dec_out})
                # {self.enc_inp: enc_inp, self.dec_inp: dec_inp, self.dec_out: dec_out,
                #  {'placeholder/x_enc_inp:0':x_enc_inp, 'placeholder/x_dec_inp:0':x_dec_inp, 'placeholder/x_dec_out:0':x_dec_out,
                #  'placeholder/y_enc_inp:0':y_enc_inp, 'placeholder/y_dec_inp:0':y_dec_inp, 'placeholder/y_dec_out:0':y_dec_out})
        return {'summaries': summaries, 'loss': loss, 'nll_loss': nll_loss,
                'kl_w': kl_w, 'kl_loss': kl_loss, 'step': step}


    def reconstruct(self, sess, sentence, sentence_dropped, sentence_feeddict=None):
        if self.prefix == "decoder":
            idx2word = self.params['idx2word']
        elif self.prefix == "encoder":
            idx2word = self.params['idx2token']
        infos = ""
        infos += 'I: %s\n' % ' '.join([idx2word[idx] for idx in sentence if idx != PAD_TOKEN])
        infos += 'D: %s\n' % ' '.join([idx2word[idx] for idx in sentence_dropped if idx != PAD_TOKEN])
        if self.isContext or self.isPointer:
            predicted_ids = sess.run(self.predicted_ids, sentence_feeddict)[0]
        else:
            predicted_ids = sess.run(self.predicted_ids, {self.enc_inp: np.atleast_2d(sentence)})[0]
        infos += 'O: %s\n' % ' '.join([idx2word[idx] for idx in predicted_ids if idx != PAD_TOKEN])
        infos += '-'*12 + "\n"
        return infos
    
    def get_new_w(self, w):
        idx = self.params['word2idx'][w]
        return idx if idx < self.params['vocab_size'] else self.params['word2idx'][UNK_STRING]

    def customized_reconstruct(self, sess, sentence):
        idx2word = self.params['idx2word']
        sentence = [self.get_new_w(w) for w in sentence.split()][:self.params['max_len']]
        print('I: %s' % ' '.join([idx2word[idx] for idx in sentence]))
        print()
        sentence = sentence + [self.params['word2idx'][PAD_STRING]] * (self.params['max_len']-len(sentence))
        predicted_ids = sess.run(self.predicted_ids, {self.enc_inp: np.atleast_2d(sentence)})[0]
        print('O: %s' % ' '.join([idx2word[idx] for idx in predicted_ids]))
        print('-'*12)

    def point_reconstruct(self, sess, sentence):
        idx2word = self.params['idx2word']
        sentence = [self.get_new_w(w) for w in sentence.split()][:self.params['max_len']]
        print('I: %s' % ' '.join([idx2word[idx] for idx in sentence]))
        print()
        sentence = sentence + [self.params['word2idx'][PAD_STRING]] * (self.params['max_len']-len(sentence))
        predicted_ids, predict_z = sess.run([self.predicted_ids, self.z], {self.enc_inp: np.atleast_2d(sentence)})
        predicted_ids = predicted_ids[0]
        print('O: %s' % ' '.join([idx2word[idx] for idx in predicted_ids]))
        print('-'*12)
        return predict_z

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

    def generate(self, sess):
        predicted_ids = sess.run(self.predicted_ids,
                                {self._batch_size: 1,
                                 self.z: np.random.randn(1, args.latent_size),
                                 self.enc_seq_len: [self.params['max_len']]})[0]
        print('G: %s' % ' '.join([self.params['idx2word'][idx] for idx in predicted_ids]))
        print('-'*12)

    def generate_byz(self, sess, z):
        predicted_ids = sess.run(self.predicted_ids,
                                {self._batch_size: 1,
                                 self.z: z,
                                 self.enc_seq_len: [self.params['max_len']]})[0]
        print('G: %s' % ' '.join([self.params['idx2word'][idx] for idx in predicted_ids]))
        print('-'*12)