from tensorflow.python.ops import array_ops
from tensorflow.contrib.seq2seq.python.ops.basic_decoder import BasicDecoder, BasicDecoderOutput
from tensorflow.contrib.seq2seq.python.ops.beam_search_decoder import BeamSearchDecoder, BeamSearchDecoderOutput

import tensorflow as tf

class ModifiedBasicDecoder(BasicDecoder):
    """
    https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/contrib/seq2seq/python/ops/basic_decoder.py
    """
    def __init__(self, cell, helper, initial_state, concat_z, output_layer=None):
        super().__init__(cell, helper, initial_state, output_layer)
        self.z = concat_z

    def initialize(self, name=None):
        (finished, first_inputs, initial_state) = super().initialize(name)
        first_inputs = array_ops.concat([first_inputs, self.z], -1)
        return (finished, first_inputs, initial_state)

    def step(self, time, inputs, state, name=None):
        (outputs, next_state, next_inputs, finished) = super().step(
            time, inputs, state, name)
        next_inputs = array_ops.concat([next_inputs, self.z], -1)
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
                 output_layer=None,
                 length_penalty_weight=0.0):
        super().__init__(cell, embedding, start_tokens, end_token, initial_state, beam_width, output_layer, length_penalty_weight)
        self.z = concat_z

    def initialize(self, name=None):
        (finished, start_inputs, initial_state) = super().initialize(name)
        start_inputs = array_ops.concat([start_inputs, self.z], -1)
        return (finished, start_inputs, initial_state)

    def step(self, time, inputs, state, name=None):
        (beam_search_output, beam_search_state, next_inputs, finished) = super().step(
            time, inputs, state, name)
        next_inputs = array_ops.concat([next_inputs, self.z], -1)
        return (beam_search_output, beam_search_state, next_inputs, finished)

class ModifiedContextDecoder(BasicDecoder):
    def __init__(self, cell, helper, initial_state, concat_z, encoder_ouputs, output_layer=None):
        super().__init__(cell, helper, initial_state, output_layer)
        self.z = concat_z
        self.encoder_ouputs = encoder_ouputs # b x t x e

    def initialize(self, name=None):
        (finished, first_inputs, initial_state) = super().initialize(name)
        first_inputs = array_ops.concat([first_inputs, self.z, self.z], -1)
        return (finished, first_inputs, initial_state)

    def step(self, time, inputs, state, name=None):
        (outputs, next_state, next_inputs, finished) = super().step(
            time, inputs, state, name)

        ### context vector
        state = next_state
        K = tf.expand_dims(state, axis=1) # b x 1 x e
        Q = self.encoder_ouputs
        V = self.encoder_ouputs
        attens = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # bxtxc bxcx1 => bxtx1 
        attens = tf.nn.softmax(attens, axis=1) # bxtx1
        context_vec = tf.reduce_sum(attens*V, axis=1)  #  bxtxc => bxc
        next_inputs = array_ops.concat([next_inputs, context_vec, self.z], -1) # bx[e+c+c]=bx640
        ### end context vector

        ### outputs_merged: state, attens
        attens = tf.squeeze(attens, [2]) # bxt
        outputs_merged = array_ops.concat([state, attens], -1) # bx[c+t]=bx656
        outputs = BasicDecoderOutput(outputs_merged, outputs[1])
        ### end outputs_merged

        return (outputs, next_state, next_inputs, finished)


class ModifiedBeamSearchContextDecoder(BeamSearchDecoder):
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
        start_inputs = array_ops.concat([start_inputs, self.z, self.z], -1)
        return (finished, start_inputs, initial_state)

    def step(self, time, inputs, state, name=None):
        (beam_search_output, beam_search_state, next_inputs, finished) = super().step(
            time, inputs, state, name)

        ### context vector
        state = beam_search_state[0]
        K = state
        Q = self.encoder_ouputs
        V = self.encoder_ouputs
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # bxtxc bxcxbeam => bxtxbeam
        attens = tf.nn.softmax(outputs, axis=1) # bxtxbeam
        context_vec = tf.expand_dims(attens, 3)*tf.expand_dims(V, 2) # bxtxbeamx1 bxtx1xc => bxtxbeamxc
        context_vec = tf.reduce_sum(context_vec, axis=1)  #  bxtxbeamxc => bxbeamxc
        next_inputs = array_ops.concat([next_inputs, context_vec, self.z], -1) # bxbeamx[e+c+c]=bx5x640
        ### end context vector

        ### outputs_merged: state, attens
        attens = tf.transpose(attens, [0,2,1]) # bxbeamxt
        outputs_merged = array_ops.concat([state, attens], -1) # bxbeamx[c+t]=bx5x656
        outputs = BeamSearchDecoderOutput(outputs_merged, beam_search_output[1], beam_search_output[2])
        ### end outputs_merged
        
        return (beam_search_output, beam_search_state, next_inputs, finished)


