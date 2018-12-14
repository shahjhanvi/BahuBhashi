import os
import numpy as np

import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.python.util import nest

import nmt_model_utils


class NMT():

    def __init__(self,
                 word2ind_1,
                 ind2word_1,
                 word2ind_2,
                 ind2word_2,
                 save_path,
                 mode='TRAIN',
                 num_layers_encoder=None,
                 num_layers_decoder=None,
                 embedding_dim=300,
                 rnn_size_encoder=256,
                 rnn_size_decoder=256,
                 keep_probability=0.8,
                 batch_size=64,
                 beam_width=10,
                 epochs=20,
                 eos="</s>",
                 sos="<s>",
                 pad='<pad>',
                 use_gru=False,
                 time_major=False,
                 clip=5,
                 learning_rate=0.001,
                 learning_rate_decay=0.9,
                 learning_rate_decay_steps=100
                 ):

        self.word2ind_1 = word2ind_1
        self.word2ind_2 = word2ind_2
        self.ind2word_1 = ind2word_1
        self.ind2word_2 = ind2word_2
        self.save_path = save_path
        self.vocab_size_1 = len(word2ind_1)
        self.vocab_size_2 = len(word2ind_2)
        self.embedding_dim = embedding_dim
        self.mode = mode.upper()
        self.num_layers_encoder = num_layers_encoder
        self.num_layers_decoder = num_layers_decoder
        self.rnn_size_encoder = rnn_size_encoder
        self.rnn_size_decoder = rnn_size_decoder
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_decay_steps = learning_rate_decay_steps
        self.keep_probability = keep_probability
        self.batch_size = batch_size
        self.beam_width = beam_width
        self.eos = eos
        self.sos = sos
        self.time_major = time_major
        self.clip = clip
        self.pad = pad
        self.use_gru = use_gru
        self.epochs = epochs

    def build_graph(self):
        self.add_placeholders()
        self.add_embeddings()
        self.add_lookup_ops()
        self.initialize_session()
        self.add_seq2seq()
        print('Graph built.')

    def add_placeholders(self):
        self.ids_1 = tf.placeholder(tf.int32, shape=[None, None], name='ids_source')
        self.ids_2 = tf.placeholder(tf.int32, shape=[None, None], name='ids_target')
        self.sequence_lengths_1 = tf.placeholder(tf.int32, shape=[None], name='sequence_length_source')
        self.sequence_lengths_2 = tf.placeholder(tf.int32, shape=[None], name='sequence_length_target')
        self.maximum_iterations = tf.reduce_max(self.sequence_lengths_2, name='max_dec_len')

    def create_word_embedding(self, embed_name, vocab_size, embed_dim):
        """Creates a matrix in given shape - [vocab_size, embed_dim]"""
        embedding = tf.get_variable(embed_name,
                                    shape=[vocab_size, embed_dim],
                                    dtype=tf.float32)
        return embedding

    def add_embeddings(self):
        """Creates the embedding matrices for both the source and target language."""
        self.embedding_1 = self.create_word_embedding('src_embedding', self.vocab_size_1, self.embedding_dim)
        self.embedding_2 = self.create_word_embedding('trgt_embedding', self.vocab_size_2, self.embedding_dim)

    def add_lookup_ops(self):
        """Additional lookup operation for both source embedding and target embedding matrix.
        """
        self.word_embeddings_1 = tf.nn.embedding_lookup(self.embedding_1, self.ids_1, name='word_embeddings_1')
        self.word_embeddings_2 = tf.nn.embedding_lookup(self.embedding_2, self.ids_2, name='word_embeddings_2')

    def make_rnn_cell(self, rnn_size, keep_probability):
        """Creates and LSTM cell or GRU cell, optionally wrapped with dropout
        """
        if self.use_gru:
            cell = tf.nn.rnn_cell.GRUCell(rnn_size)
        else:
            cell = tf.nn.rnn_cell.LSTMCell(rnn_size, activation='relu')
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_probability)
        return cell

    def make_attention_cell(self, dec_cell, rnn_size, enc_output, lengths, alignment_history=False):
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=rnn_size,
                                                                   memory=enc_output,
                                                                   memory_sequence_length=lengths,
                                                                   # normalize= True,
                                                                   name='BahdanauAttention')
        return tf.contrib.seq2seq.AttentionWrapper(cell=dec_cell,
                                                   attention_mechanism=attention_mechanism,
                                                   attention_layer_size=None,
                                                   output_attention=False,
                                                   alignment_history=alignment_history)

    def add_seq2seq(self):
        """Creates the sequence to sequence architecture.
        """
        with tf.variable_scope('dynamic_seq2seq', dtype=tf.float32):
            # Encoder
            encoder_outputs, encoder_state = self.build_encoder()

            # Decoder
            logits, sample_id, final_context_state = self.build_decoder(encoder_outputs,
                                                                        encoder_state)
            if self.mode == 'TRAIN':

                # Loss
                loss = self.compute_loss(logits)
                self.train_loss = loss
                self.word_count = tf.reduce_sum(self.sequence_lengths_1) + tf.reduce_sum(self.sequence_lengths_2)
                self.predict_count = tf.reduce_sum(self.sequence_lengths_2)
                self.global_step = tf.Variable(0, trainable=False)

                # Optimizer
                self.learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                                self.global_step,
                                                                decay_steps=self.learning_rate_decay_steps,
                                                                decay_rate=self.learning_rate_decay,
                                                                staircase=True)
                opt = tf.train.AdamOptimizer(self.learning_rate)

                # Gradients
                if self.clip > 0:
                    grads, vs = zip(*opt.compute_gradients(self.train_loss))
                    grads, gnorm = tf.clip_by_global_norm(grads, self.clip)
                    self.train_op = opt.apply_gradients(zip(grads, vs), global_step=self.global_step)
                else:
                    self.train_op = opt.minimize(self.train_loss, global_step=self.global_step)

                # # Summary
                # self.train_summary = tf.summary.merge(
                #                      tf.summary.scalar("train_loss", self.train_loss),
                #                      ] + grad_norm_summary)

            elif self.mode == 'INFER':
                loss = None
                self.infer_logits, _, self.final_context_state, self.sample_id = logits, loss, final_context_state, sample_id
                self.sample_words = self.sample_id

    def build_encoder(self):
        """The encoder. (Multiple-) LSTM-cell on top of bidirecitonal RNN
        """
        with tf.variable_scope("encoder"):
            # MY APPROACH
            fw_cell = self.make_rnn_cell(self.rnn_size_encoder // 2, self.keep_probability)
            bw_cell = self.make_rnn_cell(self.rnn_size_encoder // 2, self.keep_probability)

            for _ in range(self.num_layers_encoder):
                (out_fw, out_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=fw_cell,
                    cell_bw=bw_cell,
                    inputs=self.word_embeddings_1,
                    sequence_length=self.sequence_lengths_1,
                    dtype=tf.float32)
                encoder_outputs = tf.concat((out_fw, out_bw), -1)

            bi_state_c = tf.concat((state_fw.c, state_bw.c), -1)
            bi_state_h = tf.concat((state_fw.h, state_bw.h), -1)
            bi_lstm_state = tf.nn.rnn_cell.LSTMStateTuple(c=bi_state_c, h=bi_state_h)
            encoder_state = tuple([bi_lstm_state] * self.num_layers_encoder)

            return encoder_outputs, encoder_state

            # # GOOGLE'S APPROACH
            # num_bi_layers = 1
            # if self.num_layers_encoder is not None:
            #     num_uni_layers = self.num_layers_encoder - num_bi_layers
            # else:
            #     num_uni_layers = 1
            #
            # fw_cell = self.make_rnn_cell(self.rnn_size_encoder, self.keep_probability)
            # bw_cell = self.make_rnn_cell(self.rnn_size_encoder, self.keep_probability)
            #
            # bi_encoder_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
            #     fw_cell,
            #     bw_cell,
            #     self.word_embeddings_1,
            #     dtype=tf.float32,
            #     sequence_length=self.sequence_lengths_1,
            # )
            #
            # bi_encoder_outputs = tf.concat(bi_encoder_outputs, -1)
            #
            # uni_cell = tf.nn.rnn_cell.MultiRNNCell(
            #     [self.make_rnn_cell(self.rnn_size_encoder, self.keep_probability) for _ in range(num_uni_layers)])
            #
            # encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            #     uni_cell,
            #     bi_encoder_outputs,
            #     dtype=tf.float32,
            #     sequence_length=self.sequence_lengths_1,
            #     time_major=self.time_major
            # )
            # encoder_state = (bi_encoder_state[1],) + (
            #     (encoder_state,) if self.num_layers_encoder is None else encoder_state)
            #
            # return encoder_outputs, encoder_state

    def build_decoder(self, encoder_outputs, encoder_state):

        sos_id_2 = tf.cast(self.word2ind_2[self.sos], tf.int32)
        eos_id_2 = tf.cast(self.word2ind_2[self.eos], tf.int32)

        self.output_layer = Dense(self.vocab_size_2, name='output_projection')

        # Decoder.
        with tf.variable_scope("decoder") as decoder_scope:

            cell, decoder_initial_state = self.build_decoder_cell(
                encoder_outputs,
                encoder_state,
                self.sequence_lengths_1)

            # Train
            if self.mode != 'INFER':

                if self.time_major:
                    target_input = tf.transpose(self.ids_2)

                # Helper
                helper = tf.contrib.seq2seq.TrainingHelper(self.word_embeddings_2,
                                                           self.sequence_lengths_2,
                                                           time_major=self.time_major)

                # Decoder
                my_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                             helper,
                                                             decoder_initial_state,
                                                             output_layer=self.output_layer)

                # Dynamic decoding
                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder,
                                                                                    output_time_major=self.time_major,
                                                                                    maximum_iterations=self.maximum_iterations,
                                                                                    swap_memory=False,
                                                                                    impute_finished=True,
                                                                                    scope=decoder_scope)

                sample_id = outputs.sample_id
                logits = outputs.rnn_output


            # Inference
            else:
                start_tokens = tf.fill([self.batch_size], sos_id_2)
                end_token = eos_id_2

                if self.beam_width > 0:
                    my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                        cell=cell,
                        embedding=self.embedding_2,
                        start_tokens=start_tokens,
                        end_token=end_token,
                        initial_state=decoder_initial_state,
                        beam_width=self.beam_width,
                        output_layer=self.output_layer,
                    )

                else:
                    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embedding_2,
                                                                      start_tokens,
                                                                      end_token)

                    my_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                                 helper,
                                                                 decoder_initial_state,
                                                                 output_layer=self.output_layer)

                # Dynamic decoding
                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    my_decoder,
                    maximum_iterations=self.maximum_iterations,
                    output_time_major=self.time_major,
                    impute_finished=False,
                    swap_memory=False,
                    scope=decoder_scope)

                if self.beam_width > 0:
                    logits = tf.no_op()
                    sample_id = outputs.predicted_ids
                else:
                    logits = outputs.rnn_output
                    sample_id = outputs.sample_id

        return logits, sample_id, final_context_state

    def build_decoder_cell(self, encoder_outputs, encoder_state,
                           sequence_lengths_1):
        memory = encoder_outputs

        if self.mode == 'INFER' and self.beam_width > 0:
            memory = tf.contrib.seq2seq.tile_batch(memory, multiplier=self.beam_width)
            encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=self.beam_width)
            sequence_lengths_1 = tf.contrib.seq2seq.tile_batch(sequence_lengths_1, multiplier=self.beam_width)
            batch_size = self.batch_size * self.beam_width

        else:
            batch_size = self.batch_size

        # MY APPROACH
        if self.num_layers_decoder is not None:
            lstm_cell = tf.nn.rnn_cell.MultiRNNCell(
                [self.make_rnn_cell(self.rnn_size_decoder, self.keep_probability) for _ in
                 range(self.num_layers_decoder)])

        else:
            lstm_cell = self.make_rnn_cell(self.rnn_size_decoder, self.keep_probability)

        cell = self.make_attention_cell(lstm_cell,
                                        self.rnn_size_decoder,
                                        memory,
                                        sequence_lengths_1)

        decoder_initial_state = cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)

        return cell, decoder_initial_state

        # # -----------------------------------------------------
        #
        # # GOOGLE'S APPROACH
        #
        # # instead of MultiRNNcell we just use a cell list and pop the very bottom cell and
        # # wrap this one with attention, then using GNMTAttentionMultiRNNCell
        # cell_list = [self.make_rnn_cell(self.rnn_size_decoder, self.keep_probability) for _ in range(self.num_layers_decoder)]
        # lstm_cell = cell_list.pop(0)
        #
        # # boolean value wheter to use it or not --> only greedy inference
        # alignment_history = (self.mode == 'INFER' and self.beam_width == 0)
        #
        # attention_cell = self.make_attention_cell(
        #     lstm_cell,
        #     self.rnn_size_decoder,
        #     memory,
        #     sequence_lengths_1,
        #     alignment_history=alignment_history)
        #
        # cell = GNMTAttentionMultiCell(attention_cell, cell_list)
        #
        # decoder_initial_state = tuple(
        #     zs.clone(cell_state=es)
        #     if isinstance(zs, tf.contrib.seq2seq.AttentionWrapperState) else es
        #     for zs, es in zip(
        #         cell.zero_state(batch_size, tf.float32), encoder_state))
        #
        # return cell, decoder_initial_state
        #
        # # ----------------------------------ddr2--------------------

    def compute_loss(self, logits):
        """ Computes the loss during optimization. """
        target_output = self.ids_2
        if self.time_major:
            target_output = tf.transpose(target_output)
        max_time = self.maximum_iterations

        target_weights = tf.sequence_mask(self.sequence_lengths_2,
                                          max_time,
                                          dtype=tf.float32,
                                          name='mask')
        if self.time_major:
            target_weights = tf.transpose(target_weights)

        loss = tf.contrib.seq2seq.sequence_loss(logits=logits,
                                                targets=target_output,
                                                weights=target_weights,
                                                average_across_timesteps=True,
                                                average_across_batch=True, )

        return loss

    def train(self, inputs, targets, restore_path=None):
        assert len(inputs) == len(targets)

        self.initialize_session()
        if restore_path is not None:
            self.restore_session(restore_path)

        best_score = np.inf
        nepoch_no_imprv = 0

        inputs = np.array(inputs)
        targets = np.array(targets)

        for epoch in range(self.epochs + 1):
            print('-------------------- Epoch {} of {} --------------------'.format(epoch, self.epochs))

            # shuffle the input data before every epoch.
            shuffle_indices = np.random.permutation(len(inputs))
            inputs = inputs[shuffle_indices]
            targets = targets[shuffle_indices]

            score = self.run_epoch(inputs, targets, epoch)

            if score <= best_score:
                nepoch_no_imprv = 0
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)
                self.saver.save(self.sess, self.save_path)
                best_score = score
                print("--- new best score ---\n\n")
            else:
                # warm up epochs for the model
                if epoch > 10:
                    nepoch_no_imprv += 1
                if nepoch_no_imprv >= 5:
                    print("- early stopping {} epochs without improvement".format(nepoch_no_imprv))
                    break

    def infer(self, inputs, restore_path, targets=None):
        self.initialize_session()
        self.restore_session(restore_path)

        prediction_ids = []
        feed, _, sequence_lengths_2 = self.get_feed_dict(inputs, trgts=targets)
        infer_logits, s_ids = self.sess.run([self.infer_logits, self.sample_words], feed_dict=feed)
        prediction_ids.append(s_ids)

        # for (inps, trgts) in nmt_model_utils.minibatches(inputs, targets, self.batch_size):
        #     feed, _, sequence_lengths_2 = self.get_feed_dict(inps, trgts=trgts)
        #     infer_logits, s_ids = self.sess.run([self.infer_logits, self.sample_words], feed_dict = feed)
        #     prediction_ids.append(s_ids)

        return prediction_ids

    def run_epoch(self, inputs, targets, epoch):

        batch_size = self.batch_size
        nbatches = (len(inputs) + batch_size - 1) // batch_size
        losses = []

        for i, (inps, trgts) in enumerate(nmt_model_utils.minibatches(inputs, targets, batch_size)):
            if inps is not None and trgts is not None:
                fd, sl, s2 = self.get_feed_dict(inps,
                                                trgts=trgts,
                                                )

                _, train_loss = self.sess.run([self.train_op, self.train_loss], feed_dict=fd)

                if i % 2 == 0 or i == (nbatches - 1):
                    print('Iteration: {} of {}\ttrain_loss: {:.4f}'.format(i, nbatches - 1, train_loss))
                losses.append(train_loss)

                # if i % 10 == 0 and self.summary_dir is not None:
                # self.file_writer.add_summary(summary, epoch*nbatches + i)
            else:
                continue

        avg_loss = self.sess.run(tf.reduce_mean(losses))
        print('Average Score for this Epoch: {}'.format(avg_loss))

        return avg_loss

    def get_feed_dict(self, inps, trgts=None):
        if self.mode != 'INFER':
            inp_ids, sequence_lengths_1 = nmt_model_utils.pad_sequences(inps, self.word2ind_1[self.pad], tail=True)

            feed = {self.ids_1: inp_ids,
                    self.sequence_lengths_1: sequence_lengths_1}

            if trgts is not None:
                trgt_ids, sequence_lengths_2 = nmt_model_utils.pad_sequences(trgts, self.word2ind_2[self.pad],
                                                                             tail=True)
                feed[self.ids_2] = trgt_ids
                feed[self.sequence_lengths_2] = sequence_lengths_2

            return feed, sequence_lengths_1, sequence_lengths_2

        else:

            inp_ids, sequence_lengths_1 = nmt_model_utils.pad_sequences(inps, self.word2ind_1[self.pad], tail=True)

            feed = {self.ids_1: inp_ids,
                    self.sequence_lengths_1: sequence_lengths_1}

            if trgts is not None:
                trgt_ids, sequence_lengths_2 = nmt_model_utils.pad_sequences(trgts,
                                                                             self.word2ind_2[self.pad],
                                                                             tail=True)

                feed[self.sequence_lengths_2] = sequence_lengths_2

            return feed, sequence_lengths_1, sequence_lengths_2

    def initialize_session(self):
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def restore_session(self, restore_path):
        print('Restore graph from ', restore_path)
        self.saver.restore(self.sess, restore_path)


# RNN cell by google for neural machine translation --> from gnmt_model.py in the nmt repository.
# inherits from the MultiRNNCell.
# I had my own, slightly different approach for encoder and decoder, but this one seems to perform better.
class GNMTAttentionMultiCell(tf.nn.rnn_cell.MultiRNNCell):
    """A MultiCell with GNMT attention style."""

    def __init__(self, attention_cell, cells, use_new_attention=False):
        """Creates a GNMTAttentionMultiCell.
        Args:
          attention_cell: An instance of AttentionWrapper.
          cells: A list of RNNCell wrapped with AttentionInputWrapper.
          use_new_attention: Whether to use the attention generated from current
            step bottom layer's output. Default is False.
        """
        cells = [attention_cell] + cells
        self.use_new_attention = use_new_attention
        super(GNMTAttentionMultiCell, self).__init__(cells, state_is_tuple=True)

    def __call__(self, inputs, state, scope=None):
        """Run the cell with bottom layer's attention copied to all upper layers."""
        if not nest.is_sequence(state):
            raise ValueError(
                "Expected state to be a tuple of length %d, but received: %s"
                % (len(self.state_size), state))

        with tf.variable_scope(scope or "multi_rnn_cell"):
            new_states = []

            with tf.variable_scope("cell_0_attention"):
                attention_cell = self._cells[0]
                attention_state = state[0]
                cur_inp, new_attention_state = attention_cell(inputs, attention_state)
                new_states.append(new_attention_state)

            for i in range(1, len(self._cells)):
                with tf.variable_scope("cell_%d" % i):

                    cell = self._cells[i]
                    cur_state = state[i]

                    if self.use_new_attention:
                        cur_inp = tf.concat([cur_inp, new_attention_state.attention], -1)
                    else:
                        cur_inp = tf.concat([cur_inp, attention_state.attention], -1)

                    cur_inp, new_state = cell(cur_inp, cur_state)
                    new_states.append(new_state)

        return cur_inp, tuple(new_states)