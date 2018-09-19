"""
This module implements a multi-layer recurrent neural network as encoder, and
an attention-based decoder to solve machine translation task.
__Author__ == 'Haowen Xu'
__Date__ == '09-11-2018'
"""
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
import math

from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops.rnn_cell import DropoutWrapper, ResidualWrapper

from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder

from tensorflow.python.layers.core import Dense
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops

from ..basic_model import Basic_model
import utils
from utils import data_utils

class RNNAttention(Basic_model):
    def __init__(self, config, sess, mode, exp_name='RNNAttention', logger=None):
        self.mode = mode.lower()

        # Config must define RNN type and attention type
        super(RNNAttention, self).__init__(config, sess, exp_name)
        self.logger = logger
        self.dtype = tf.float32
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.use_beamsearch_decode=False
        if self.mode == 'decode':
            self.use_beamsearch_decode = True if config.beam_width > 1 else False

        self._build_model()

    def _build_model(self):
        with tf.variable_scope(self.exp_name):
            self._build_placeholder()
            self._build_encoder()
            self._build_decoder()
        # Merge all the training summaries
        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver()

    def _build_placeholder(self):
        # [batch_size, max_time_steps]
        self.encoder_inputs = tf.placeholder(dtype=tf.int32, shape=(None, None),
                                             name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(
            dtype=tf.int32, shape=(None,), name='encoder_inputs_length')
        # dynamic batch_size
        self.batch_size = tf.shape(self.encoder_inputs)[0]

        if self.mode == 'train':
            self.decoder_inputs = tf.placeholder(
                dtype=tf.int32, shape=(None, None), name='decoder_inputs')
            self.decoder_inputs_length = tf.placeholder(
                dtype=tf.int32, shape=(None,), name='decoder_inputs_length')

            decoder_start_token = tf.ones(
                shape=[self.batch_size, 1], dtype=tf.int32) * data_utils.start_token
            decoder_end_token = tf.ones(
                shape=[self.batch_size, 1], dtype=tf.int32) * data_utils.end_token

            # decoder_inputs_train: [batch_size , max_time_steps + 1]
            # insert _GO symbol in front of each decoder input
            self.decoder_inputs_train = tf.concat([decoder_start_token,
                                                  self.decoder_inputs], axis=1)

            # decoder_inputs_length_train: [batch_size]
            self.decoder_inputs_length_train = self.decoder_inputs_length + 1

            self.decoder_targets_train = tf.concat([self.decoder_inputs,
                                                    decoder_end_token], axis=1)

        self.keep_prob_placeholder = tf.placeholder(self.dtype,
                                                    shape=[],
                                                    name='keep_prob')

    def _build_single_cell(self):
        config = self.config
        cell_type = LSTMCell
        if config.cell_type.lower() == 'gru':
            cell_type = GRUCell
        cell = cell_type(config.encoder_hidden_units)
        if config.use_dropout:
            cell = DropoutWrapper(cell, dtype=self.dtype,
                                  output_keep_prob=self.keep_prob_placeholder,)
        if config.use_residual:
            cell = ResidualWrapper(cell)

        return cell

    def _build_encoder_cell(self):
        config = self.config
        return MultiRNNCell([self._build_single_cell()
                             for i in range(config.encoder_depth)])

    def _build_decoder_cell(self):
        config = self.config
        encoder_outputs = self.encoder_outputs
        encoder_last_state = self.encoder_last_state
        encoder_inputs_length = self.encoder_inputs_length

        # To use BeamSearchDecoder, encoder_outputs, encoder_last_state, encoder_inputs_length
        # needs to be tiled so that: [batch_size, .., ..] -> [batch_size x beam_width, .., ..]
        if self.use_beamsearch_decode:
            print ("use beamsearch decoding..")
            encoder_outputs = seq2seq.tile_batch(
                self.encoder_outputs, multiplier=config.beam_width)
            encoder_last_state = nest.map_structure(
                lambda s: seq2seq.tile_batch(s, config.beam_width), self.encoder_last_state)
            encoder_inputs_length = seq2seq.tile_batch(
                self.encoder_inputs_length, multiplier=config.beam_width)

        # Building attention mechanism: Default Bahdanau
        # 'Luong' style attention: https://arxiv.org/abs/1508.04025
        if config.attention_type.lower() == 'luong':
            self.attention_mechanism = attention_wrapper.LuongAttention(
                num_units=config.attn_hidden_units, memory=encoder_outputs,
                memory_sequence_length=encoder_inputs_length,)
        else:
        # 'Bahdanau' style attention: https://arxiv.org/abs/1409.0473
            self.attention_mechanism = attention_wrapper.BahdanauAttention(
                num_units=config.attn_hidden_units, memory=encoder_outputs,
                memory_sequence_length=encoder_inputs_length,)

        # Building decoder_cell
        self.decoder_cell_list = [
            self._build_single_cell() for i in range(config.decoder_depth)]
        decoder_initial_state = encoder_last_state

        def attn_decoder_input_fn(inputs, attention):
            if not config.attn_input_feeding:
                return inputs

            # Essential when use_residual=True
            _input_layer = Dense(config.attn_hidden_units, dtype=self.dtype,
                                 name='attn_input_feeding')
            return _input_layer(array_ops.concat([inputs, attention], -1))

        # AttentionWrapper wraps RNNCell with the attention_mechanism
        # Note: We implement Attention mechanism only on the top decoder layer
        self.decoder_cell_list[-1] = attention_wrapper.AttentionWrapper(
            cell=self.decoder_cell_list[-1],
            attention_mechanism=self.attention_mechanism,
            attention_layer_size=config.attn_hidden_units,
            cell_input_fn=attn_decoder_input_fn,
            initial_cell_state=encoder_last_state[-1],
            alignment_history=False,
            name='Attention_Wrapper')

        # To be compatible with AttentionWrapper, the encoder last state
        # of the top layer should be converted into the AttentionWrapperState form
        # We can easily do this by calling AttentionWrapper.zero_state

        # Also if beamsearch decoding is used, the batch_size argument in .zero_state
        # should be ${decoder_beam_width} times to the origianl batch_size
        batch_size = self.batch_size if not self.use_beamsearch_decode \
                     else self.batch_size * config.beam_width
        initial_state = [state for state in encoder_last_state]

        initial_state[-1] = self.decoder_cell_list[-1].zero_state(
          batch_size=batch_size, dtype=self.dtype)
        decoder_initial_state = tuple(initial_state)

        return MultiRNNCell(self.decoder_cell_list), decoder_initial_state

    def _build_encoder(self):
        config = self.config
        logger = self.logger
        if not logger:
            logger.info('building encoder...')
        else:
            print('building encoder...')
        with tf.variable_scope('encoder'):
            # Building encoder cell
            self.encoder_cell = self._build_encoder_cell()

            # Initialize encoder_embeddings to have variance=1.
            sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3,
                                                        dtype=self.dtype)

            self.encoder_embeddings = tf.get_variable(name='embedding',
                shape=[config.num_encoder_symbols, config.embedding_size],
                initializer=initializer, dtype=self.dtype)

            # Embedded_inputs: [batch_size, time_step, embedding_size]
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(
                params=self.encoder_embeddings, ids=self.encoder_inputs)

            # Input projection layer to feed embedded inputs to the cell
            # ** Essential when use_residual=True to match input/output dims
            input_layer = Dense(config.encoder_hidden_units, dtype=self.dtype,
                                name='input_projection')

            # Embedded inputs having gone through input projection layer
            self.encoder_inputs_embedded = input_layer(self.encoder_inputs_embedded)

            # Encode input sequences into context vectors:
            # encoder_outputs: [batch_size, max_time_step, cell_output_size]
            # encoder_state: [batch_size, cell_output_size]
            self.encoder_outputs, self.encoder_last_state = tf.nn.dynamic_rnn(
                cell=self.encoder_cell, inputs=self.encoder_inputs_embedded,
                sequence_length=self.encoder_inputs_length, dtype=self.dtype,
                time_major=False)

    def _build_decoder(self):
        config = self.config
        logger = self.logger
        if not logger:
            logger.info('building decoder and attention...')
        else:
            print('building decoder and attention...')
        with tf.variable_scope('decoder'):
            # Building decoder_cell and decoder_initial_state
            self.decoder_cell, self.decoder_initial_state = self._build_decoder_cell()

            # Initialize decoder embeddings to have variance=1.
            sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype=self.dtype)

            self.decoder_embeddings = tf.get_variable(name='embedding',
                shape=[config.num_decoder_symbols, config.embedding_size],
                initializer=initializer, dtype=self.dtype)

            # Input projection layer to feed embedded inputs to the cell
            # ** Essential when use_residual=True to match input/output dims
            input_layer = Dense(config.decoder_hidden_units,
                                dtype=self.dtype, name='input_projection')

            # Output projection layer to convert cell_outputs to logits
            output_layer = Dense(config.num_decoder_symbols, name='output_projection')

            if self.mode == 'train':
                # decoder_inputs_embedded: [batch_size, max_time_step + 1,
                # embedding_size]
                self.decoder_inputs_embedded = tf.nn.embedding_lookup(
                    params=self.decoder_embeddings, ids=self.decoder_inputs_train)

                # Embedded inputs having gone through input projection layer
                self.decoder_inputs_embedded = input_layer(
                    self.decoder_inputs_embedded)

                # Helper to feed inputs for training: read inputs from dense ground truth vectors
                training_helper = seq2seq.TrainingHelper(inputs=self.decoder_inputs_embedded,
                                                   sequence_length=self.decoder_inputs_length_train,
                                                   time_major=False,
                                                   name='training_helper')

                training_decoder = seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                   helper=training_helper,
                                                   initial_state=self.decoder_initial_state,
                                                   output_layer=output_layer)

                # Maximum decoder time_steps in current batch
                max_decoder_length = tf.reduce_max(self.decoder_inputs_length_train)

                # decoder_outputs_train: BasicDecoderOutput
                #                        namedtuple(rnn_outputs, sample_id)
                # decoder_outputs_train.rnn_output: [batch_size, max_time_step + 1, num_decoder_symbols] if output_time_major=False
                #                                   [max_time_step + 1, batch_size, num_decoder_symbols] if output_time_major=True
                # decoder_outputs_train.sample_id: [batch_size], tf.int32
                (self.decoder_outputs_train, self.decoder_last_state_train,
                 self.decoder_outputs_length_train) = (seq2seq.dynamic_decode(
                    decoder=training_decoder,
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=max_decoder_length))

                # More efficient to do the projection on the batch-time-concatenated tensor
                # logits_train: [batch_size, max_time_step + 1, num_decoder_symbols]
                # self.decoder_logits_train = output_layer(self.decoder_outputs_train.rnn_output)
                self.decoder_logits_train = tf.identity(self.decoder_outputs_train.rnn_output)
                # Use argmax to extract decoder symbols to emit
                self.decoder_pred_train = tf.argmax(self.decoder_logits_train, axis=-1,
                                                    name='decoder_pred_train')

                # masks: masking for valid and padded time steps, [batch_size, max_time_step + 1]
                masks = tf.sequence_mask(lengths=self.decoder_inputs_length_train,
                                         maxlen=max_decoder_length, dtype=self.dtype, name='masks')

                # Computes per word average cross-entropy over a batch
                # Internally calls 'nn_ops.sparse_softmax_cross_entropy_with_logits' by default
                self.loss = seq2seq.sequence_loss(logits=self.decoder_logits_train,
                                                  targets=self.decoder_targets_train,
                                                  weights=masks,
                                                  average_across_timesteps=True,
                                                  average_across_batch=True,)

                # Computes per word accuracy over a batch
                self.decoder_predicts_train = tf.argmax(self.decoder_logits_train, axis=-1)
                self.decoder_predicts_train = tf.cast(self.decoder_predicts_train, tf.int32)
                correct_prediction = tf.equal(self.decoder_predicts_train,
                                              self.decoder_targets_train)
                correct_prediction = tf.cast(correct_prediction, tf.float32)
                masked_sum = tf.reduce_sum(correct_prediction * masks)
                self.acc = masked_sum / tf.reduce_sum(masks)

                # Training summary for the current batch_loss

                tf.summary.scalar('loss', self.loss)

                # Contruct graphs for minimizing loss
                self.init_optimizer()

            elif self.mode == 'decode':

                # Start_tokens: [batch_size,] `int32` vector
                start_tokens = tf.ones([self.batch_size,], tf.int32) * data_utils.start_token
                end_token = data_utils.end_token

                def embed_and_input_proj(inputs):
                    return input_layer(tf.nn.embedding_lookup(self.decoder_embeddings, inputs))

                if not self.use_beamsearch_decode:
                    # Helper to feed inputs for greedy decoding: uses the argmax of the output
                    decoding_helper = seq2seq.GreedyEmbeddingHelper(start_tokens=start_tokens,
                                                                    end_token=end_token,
                                                                    embedding=embed_and_input_proj)
                    # Basic decoder performs greedy decoding at each time step
                    print("building greedy decoder..")
                    inference_decoder = seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                             helper=decoding_helper,
                                                             initial_state=self.decoder_initial_state,
                                                             output_layer=output_layer)
                else:
                    # Beamsearch is used to approximately find the most likely translation
                    print("building beamsearch decoder..")
                    inference_decoder = beam_search_decoder.BeamSearchDecoder(cell=self.decoder_cell,
                                                               embedding=embed_and_input_proj,
                                                               start_tokens=start_tokens,
                                                               end_token=end_token,
                                                               initial_state=self.decoder_initial_state,
                                                               beam_width=config.beam_width,
                                                               output_layer=output_layer,)
                # For GreedyDecoder, return
                # decoder_outputs_decode: BasicDecoderOutput instance
                #                         namedtuple(rnn_outputs, sample_id)
                # decoder_outputs_decode.rnn_output: [batch_size, max_time_step, num_decoder_symbols] 	if output_time_major=False
                #                                    [max_time_step, batch_size, num_decoder_symbols] 	if output_time_major=True
                # decoder_outputs_decode.sample_id: [batch_size, max_time_step], tf.int32		if output_time_major=False
                #                                   [max_time_step, batch_size], tf.int32               if output_time_major=True

                # For BeamSearchDecoder, return
                # decoder_outputs_decode: FinalBeamSearchDecoderOutput instance
                #                         namedtuple(predicted_ids, beam_search_decoder_output)
                # decoder_outputs_decode.predicted_ids: [batch_size, max_time_step, beam_width] if output_time_major=False
                #                                       [max_time_step, batch_size, beam_width] if output_time_major=True
                # decoder_outputs_decode.beam_search_decoder_output: BeamSearchDecoderOutput instance
                #                                                    namedtuple(scores, predicted_ids, parent_ids)

                (self.decoder_outputs_decode, self.decoder_last_state_decode,
                 self.decoder_outputs_length_decode) = (seq2seq.dynamic_decode(
                    decoder=inference_decoder,
                    output_time_major=False,
                    #impute_finished=True,	# error occurs
                    maximum_iterations=config.max_decode_step))

                if not self.use_beamsearch_decode:
                    # decoder_outputs_decode.sample_id: [batch_size, max_time_step]
                    # Or use argmax to find decoder symbols to emit:
                    # self.decoder_pred_decode = tf.argmax(self.decoder_outputs_decode.rnn_output,
                    #                                      axis=-1, name='decoder_pred_decode')

                    # Here, we use expand_dims to be compatible with the result of the beamsearch decoder
                    # decoder_pred_decode: [batch_size, max_time_step, 1] (output_major=False)
                    self.decoder_pred_decode = tf.expand_dims(self.decoder_outputs_decode.sample_id, -1)

                else:
                    # Use beam search to approximately find the most likely translation
                    # decoder_pred_decode: [batch_size, max_time_step, beam_width] (output_major=False)
                    self.decoder_pred_decode = self.decoder_outputs_decode.predicted_ids

    def init_optimizer(self):
        config = self.config
        print("setting optimizer..")
        # Gradients and SGD update operation for training the model
        trainable_params = tf.trainable_variables()
        if config.optimizer.lower() == 'adadelta':
            self.opt = tf.train.AdadeltaOptimizer(learning_rate=config.learning_rate)
        elif config.optimizer.lower() == 'adam':
            self.opt = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
        elif config.optimizer.lower() == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(learning_rate=config.learning_rate)
        else:
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=config.learning_rate)

        # Compute gradients of loss w.r.t. all trainable variables
        gradients = tf.gradients(self.loss, trainable_params)

        # Clip gradients by a given maximum_gradient_norm
        clip_gradients, _ = tf.clip_by_global_norm(gradients, config.max_gradient_norm)

        # Update the model
        self.updates = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=self.global_step)

    def build_single_cell(self):
        cell_type = LSTMCell
        if (self.cell_type.lower() == 'gru'):
            cell_type = GRUCell
        cell = cell_type(self.hidden_units)

        if self.use_dropout:
            cell = DropoutWrapper(cell, dtype=self.dtype,
                                  output_keep_prob=self.keep_prob_placeholder,)
        if self.use_residual:
            cell = ResidualWrapper(cell)

        return cell

    def train(self, encoder_inputs, encoder_inputs_length,
              decoder_inputs, decoder_inputs_length,
              return_acc=False):
        """Run a train step of the model feeding the given inputs.

        Args:
          encoder_inputs: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder inputs
          encoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch
          decoder_inputs: a numpy int matrix of [batch_size, max_target_time_steps]
              to feed as decoder inputs
          decoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch

        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.
        """
        # Check if the model is 'training' mode
        if self.mode.lower() != 'train':
            raise ValueError("train step can only be operated in train mode")

        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length,
                                      decoder_inputs, decoder_inputs_length, False)
        # Input feeds for dropout
        input_feed[self.keep_prob_placeholder.name] = self.config.keep_prob

        if not return_acc:
            output_feed = [self.updates,	# Update Op that does optimization
                           self.loss,	# Loss for current batch
                           self.summary_op]	# Training summary

            outputs = self.sess.run(output_feed, input_feed)
            return outputs[1], outputs[2]	# loss, summary
        else:
            output_feed = [self.updates,	# Update Op that does optimization
                           self.loss,	# Loss for current batch
                           self.acc,   # Accuracy for current batch
                           self.summary_op]	# Training summary

            outputs = self.sess.run(output_feed, input_feed)
            return outputs[1], outputs[2], outputs[3]

    def eval(self, encoder_inputs, encoder_inputs_length,
             decoder_inputs, decoder_inputs_length,
             return_acc=False):
        """Run a evaluation step of the model feeding the given inputs.

        Args:
          encoder_inputs: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder inputs
          encoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch
          decoder_inputs: a numpy int matrix of [batch_size, max_target_time_steps]
              to feed as decoder inputs
          decoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch

        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.
        """

        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length,
                                      decoder_inputs, decoder_inputs_length, False)
        # Input feeds for dropout
        input_feed[self.keep_prob_placeholder.name] = 1.0

        if not return_acc:
            output_feed = [self.loss,	# Loss for current batch
                           self.summary_op]	# Training summary
            outputs = self.sess.run(output_feed, input_feed)
            return outputs[0], outputs[1]	# loss
        else:
            output_feed = [self.loss,	# Loss for current batch
                           self.acc,    # Accuracy for current batch
                           self.summary_op]	# Training summary
            outputs = self.sess.run(output_feed, input_feed)
            return outputs[0], outputs[1], outputs[2]	# loss

    def inference(self, encoder_inputs, encoder_inputs_length):
        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length,
                                      decoder_inputs=None, decoder_inputs_length=None,
                                      decode=True)
        input_feed[self.keep_prob_placeholder.name] = 1.0
        output_feed = [self.decoder_pred_decode]
        outputs = self.sess.run(output_feed, input_feed)
        return outputs[0]

    def init_parameters(self):
        sess = self.sess
        sess.run(tf.global_variables_initializer())

    def check_feeds(self, encoder_inputs, encoder_inputs_length,
                    decoder_inputs, decoder_inputs_length, decode):
        """
        Args:
          encoder_inputs: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder inputs
          encoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch
          decoder_inputs: a numpy int matrix of [batch_size, max_target_time_steps]
              to feed as decoder inputs
          decoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch
          decode: a scalar boolean that indicates decode mode
        Returns:
          A feed for the model that consists of encoder_inputs, encoder_inputs_length,
          decoder_inputs, decoder_inputs_length
        """

        input_batch_size = encoder_inputs.shape[0]
        if input_batch_size != encoder_inputs_length.shape[0]:
            raise ValueError("Encoder inputs and their lengths must be equal in their "
                "batch_size, %d != %d" % (input_batch_size, encoder_inputs_length.shape[0]))

        if not decode:
            target_batch_size = decoder_inputs.shape[0]
            if target_batch_size != input_batch_size:
                raise ValueError("Encoder inputs and Decoder inputs must be equal in their "
                    "batch_size, %d != %d" % (input_batch_size, target_batch_size))
            if target_batch_size != decoder_inputs_length.shape[0]:
                raise ValueError("Decoder targets and their lengths must be equal in their "
                    "batch_size, %d != %d" % (target_batch_size, decoder_inputs_length.shape[0]))

        input_feed = {}

        input_feed[self.encoder_inputs.name] = encoder_inputs
        input_feed[self.encoder_inputs_length.name] = encoder_inputs_length

        if not decode:
            input_feed[self.decoder_inputs.name] = decoder_inputs
            input_feed[self.decoder_inputs_length.name] = decoder_inputs_length

        return input_feed

