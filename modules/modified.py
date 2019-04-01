from tensorflow.python.ops import array_ops
from tensorflow.contrib.seq2seq.python.ops.basic_decoder import BasicDecoder, BasicDecoderOutput
from tensorflow.contrib.seq2seq.python.ops.beam_search_decoder import BeamSearchDecoder, BeamSearchDecoderOutput, _beam_search_step
from modules.beam_search_decoder import BeamSearchDecoder as RewriteBeamSearchDecoder
from modules.beam_search_decoder import BeamSearchDecoderOutput as RewriteBeamSearchDecoderOutput
from modules.beam_search_decoder import _beam_search_step as _rewrite_beam_search_step

from tensorflow.python.util import nest
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops

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

class ContextDecoder(BasicDecoder):
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

        ### context vector
        state = next_state
        K = tf.expand_dims(state, axis=1) # b x 1 x e
        Q = self.encoder_ouputs
        V = self.encoder_ouputs
        attens = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # bxtxc bxcx1 => bxtx1 
        attens = tf.nn.softmax(attens, axis=1) # bxtx1
        context_vec = tf.reduce_sum(attens*V, axis=1)  #  bxtxc => bxc
        next_inputs = array_ops.concat([next_inputs, self.z], -1) # bx[e+c]
        ### end context vector

        ### outputs_merged: state, context
        outputs_merged = array_ops.concat([state, context_vec], -1) # bx[c+c]
        outputs = BasicDecoderOutput(outputs_merged, outputs[1])
        ## BasicDecoderOutput - rnn_output, sample_id
        ### end outputs_merged

        return (outputs, next_state, next_inputs, finished)

class ContextBeamSearchDecoder(BeamSearchDecoder):
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
        batch_size = self._batch_size
        beam_width = self._beam_width
        end_token = self._end_token
        length_penalty_weight = self._length_penalty_weight

        with ops.name_scope(name, "BeamSearchDecoderStep", (time, inputs, state)):
            cell_state = state.cell_state
            inputs = nest.map_structure(
                lambda inp: self._merge_batch_beams(inp, s=inp.shape[2:]), inputs)
            cell_state = nest.map_structure(
                self._maybe_merge_batch_beams,
                cell_state, self._cell.state_size)
            cell_outputs, next_cell_state = self._cell(inputs, cell_state)
            cell_outputs = nest.map_structure(
                lambda out: self._split_batch_beams(out, out.shape[1:]), cell_outputs)
            next_cell_state = nest.map_structure(
                self._maybe_split_batch_beams,
                next_cell_state, self._cell.state_size)

            ### context vector
            K = next_cell_state
            Q = self.encoder_ouputs
            V = self.encoder_ouputs
            outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # bxtxc bxcxbeam => bxtxbeam
            attens = tf.nn.softmax(outputs, axis=1) # bxtxbeam
            context_vec = tf.expand_dims(attens, 3)*tf.expand_dims(V, 2) # bxtxbeamx1 bxtx1xc => bxtxbeamxc
            context_vec = tf.reduce_sum(context_vec, axis=1)  #  bxtxbeamxc => bxbeamxc
            ### end context vector
            ### cell_outputs vector
            cell_outputs = array_ops.concat([cell_outputs, context_vec], -1)
            ### end cell_outputs vector

            if self._output_layer is not None:
                cell_outputs = self._output_layer(cell_outputs)

            beam_search_output, beam_search_state = _beam_search_step(
                time=time,
                logits=cell_outputs,
                next_cell_state=next_cell_state,
                beam_state=state,
                batch_size=batch_size,
                beam_width=beam_width,
                end_token=end_token,
                length_penalty_weight=length_penalty_weight)

            finished = beam_search_state.finished
            sample_ids = beam_search_output.predicted_ids
            next_inputs = control_flow_ops.cond(
                math_ops.reduce_all(finished), lambda: self._start_inputs,
                lambda: self._embedding_fn(sample_ids))

            ### next_inputs vector
            next_inputs = array_ops.concat([next_inputs, self.z], -1) # bxbeamx[e+c+c]=bx5x640
            ### next_inputs vector
        
        return (beam_search_output, beam_search_state, next_inputs, finished)

class PointerDecoder(BasicDecoder):
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

        ### context vector
        state = next_state
        K = tf.expand_dims(state, axis=1) # b x 1 x e
        Q = self.encoder_ouputs
        V = self.encoder_ouputs
        attens = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # bxtxc bxcx1 => bxtx1 
        attens = tf.nn.softmax(attens, axis=1) # bxtx1
        context_vec = tf.reduce_sum(attens*V, axis=1)  #  bxtxc => bxc
        next_inputs = array_ops.concat([next_inputs, self.z], -1) # bx[e+c+c]=bx640
        ### end context vector

        ### outputs_merged: state, attens
        attens = tf.squeeze(attens, [2]) # bxt
        outputs_merged = array_ops.concat([state, context_vec, attens], -1) # bx[c+c+t]=bx656
        outputs = BasicDecoderOutput(outputs_merged, outputs[1])
        ### end outputs_merged

        return (outputs, next_state, next_inputs, finished)


class PointerBeamSearchDecoder(RewriteBeamSearchDecoder):
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
                 pointer_layer=None,
                 length_penalty_weight=0.0):
        super().__init__(cell, embedding, start_tokens, end_token, initial_state, beam_width, output_layer, length_penalty_weight)
        self.z = concat_z
        self.encoder_ouputs = encoder_ouputs # b x t x e
        self.pointer_layer = pointer_layer

    def initialize(self, name=None):
        (finished, start_inputs, initial_state) = super().initialize(name)
        start_inputs = array_ops.concat([start_inputs, self.z], -1)
        return (finished, start_inputs, initial_state)

    def step(self, time, inputs, state, name=None):
        batch_size = self._batch_size
        beam_width = self._beam_width
        end_token = self._end_token
        length_penalty_weight = self._length_penalty_weight

        with ops.name_scope(name, "BeamSearchDecoderStep", (time, inputs, state)):
            cell_state = state.cell_state
            inputs = nest.map_structure(
                lambda inp: self._merge_batch_beams(inp, s=inp.shape[2:]), inputs)
            cell_state = nest.map_structure(
                self._maybe_merge_batch_beams,
                cell_state, self._cell.state_size)
            cell_outputs, next_cell_state = self._cell(inputs, cell_state)
            cell_outputs = nest.map_structure(
                lambda out: self._split_batch_beams(out, out.shape[1:]), cell_outputs)
            next_cell_state = nest.map_structure(
                self._maybe_split_batch_beams,
                next_cell_state, self._cell.state_size)

            ### context vector
            K = next_cell_state
            Q = self.encoder_ouputs
            V = self.encoder_ouputs
            outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # bxtxc bxcxbeam => bxtxbeam
            attens = tf.nn.softmax(outputs, axis=1) # bxtxbeam
            context_vec = tf.expand_dims(attens, 3)*tf.expand_dims(V, 2) # bxtxbeamx1 bxtx1xc => bxtxbeamxc
            context_vec = tf.reduce_sum(context_vec, axis=1)  #  bxtxbeamxc => bxbeamxc
            ### end context vector
            ### cell_outputs vector
            cell_outputs = array_ops.concat([cell_outputs, context_vec], -1)
            ### end cell_outputs vector

            if self._output_layer is not None:
                cell_outputs = self._output_layer(cell_outputs)

            beam_search_output, beam_search_state = _rewrite_beam_search_step(
                time=time,
                logits=cell_outputs,
                next_cell_state=next_cell_state,
                beam_state=state,
                batch_size=batch_size,
                beam_width=beam_width,
                end_token=end_token,
                length_penalty_weight=length_penalty_weight)

            finished = beam_search_state.finished
            sample_ids = beam_search_output.predicted_ids
            next_inputs = control_flow_ops.cond(
                math_ops.reduce_all(finished), lambda: self._start_inputs,
                lambda: self._embedding_fn(sample_ids))

            ### next_inputs vector
            next_inputs = array_ops.concat([next_inputs, self.z], -1) # bxbeamx[e+c+c]=bx5x640

            attens = tf.transpose(attens, [0,2,1]) # bxbeamxt
            pointer = self.pointer_layer.apply(next_cell_state) # bxbeamx1
            outputs_merged = array_ops.concat([attens, pointer], -1) # bxbeamx[c+t]=bx5x656
            beam_search_output = RewriteBeamSearchDecoderOutput(outputs_merged, beam_search_output[1], beam_search_output[2], beam_search_output[3])
            ### next_inputs vector
        
        return (beam_search_output, beam_search_state, next_inputs, finished)


