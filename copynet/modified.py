from tensorflow.python.ops import array_ops
from tensorflow.contrib.seq2seq.python.ops.basic_decoder import BasicDecoder
from tensorflow.contrib.seq2seq.python.ops.beam_search_decoder import BeamSearchDecoder
import tensorflow as tf

class ModifiedBasicDecoder(BasicDecoder):
    """
    https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/contrib/seq2seq/python/ops/basic_decoder.py
    """
    def __init__(self, cell, helper, initial_state, concat_z, encoder_ouputs, output_layer=None):
        super().__init__(cell, helper, initial_state, output_layer)
        self.z = concat_z
        self.encoder_ouputs = encoder_ouputs # b x t x e

    def initialize(self, name=None):
        (finished, first_inputs, initial_state) = super().initialize(name)
        first_inputs = array_ops.concat([first_inputs, self.z], -1)
        return (finished, first_inputs, initial_state)

    def step(self, time, inputs, state, name=None):
        (outputs, next_state, next_inputs, finished) = super().step(
            time, inputs, state, name)

        state_exp = tf.expand_dims(state, axis=1) # b x 1 x e
        logits = tf.reduce_sum(state_exp * self.encoder_ouputs, axis=2) # b x t
        attens = tf.expand_dims(tf.nn.softmax(logits), axis=2) # b x t x 1
        context_vec = tf.reduce_sum(attens * self.encoder_ouputs, axis=1) # b x e
        next_inputs = array_ops.concat([next_inputs, context_vec], -1)
        return (outputs, next_state, next_inputs, finished)


class ModifiedBeamSearchDecoder(BeamSearchDecoder):
    def __init__(self,
                 cell,
                 embedding,
                 start_tokens,
                 end_token,
                 initial_state,
                 beam_width,
                 concat_z,
                 encoder_ouputs,
                 output_layer=None,
                 length_penalty_weight=0.0):
        super().__init__(cell, embedding, start_tokens, end_token, initial_state, beam_width, output_layer, length_penalty_weight)
        self.z = concat_z
        self.encoder_ouputs = encoder_ouputs # b x t x e

    def initialize(self, name=None):
        (finished, start_inputs, initial_state) = super().initialize(name)
        start_inputs = array_ops.concat([start_inputs, self.z], -1)
        return (finished, start_inputs, initial_state)

    def step(self, time, inputs, state, name=None):
        (beam_search_output, beam_search_state, next_inputs, finished) = super().step(
            time, inputs, state, name)

        print("state:", state)
        print("state[0]:", state[0])
        state_exp = tf.expand_dims(state[0], axis=1) # b x beam x e => b x 1 x beam x  e
        logits = tf.reduce_sum(state_exp * self.encoder_ouputs, axis=2) # b x t
        attens = tf.expand_dims(tf.nn.softmax(logits), axis=2) # b x t x 1
        context_vec = tf.reduce_sum(attens * self.encoder_ouputs, axis=1) # b x e
        next_inputs = array_ops.concat([next_inputs, context_vec], -1)
        # next_inputs = array_ops.concat([next_inputs, self.z], -1)
        return (beam_search_output, beam_search_state, next_inputs, finished)