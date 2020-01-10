import tensorflow as tf
from tensorflow.keras.layers import *

assert tf.__version__=="2.0.0", f"Expect TF-2.0.0 but get {tf.__version__}"

class multiplicative_attention_model(tf.keras.Model):
	"""
	A Tensorflow 2.0 implementation of encoder-decoder attention model as illustrated in Effective Approaches to Attention-based Neural Machine Translation.
	The following types of attentions are implemented:
	• Global attention with dot alignment
	• Local attention with monotonic alignment
	
	# Arguments:

		attn_mode: `global` or `local`
		input_shape: shape of input data, should be a two-dimensional tuple (batch, n_step)
		input_vocab_size: size of input vocabulary size excluding <EOS>
		output_vocab_size: size of output vocabulary size excluding <EOS>
		input_embed_dim: word embedding dimension for input
		hidden_state_dim: hidden state dimension for encoder and decoder LSTM
		local_attn_window: the context vector will be extracted from window [current_position ± local_attn_window]
		enc_lstm_dropout: fraction of the encoder lstm units to drop
		dec_lstm_dropout: fraction of the decoder lstm units to drop
		name: name of the model

	# Paper:

		https://www-nlp.stanford.edu/pubs/emnlp15_attn.pdf

	# Examples

	```python
		# Expected shape for input data and output data
		attn_mode = "global"
		input_shape = (None,40)
		input_vocab_size = 200
		output_vocab_size = 300

		# Initialize Model
		model = multiplicative_attention_model(attn_mode, input_shape, input_vocab_size, output_vocab_size)

		# Compile Model
		learning_rate = 1e-3
		model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
					  optimizer=tf.keras.optimizers.Adam(learning_rate),
					  metrics=[tf.keras.metrics.sparse_categorical_accuracy]
					  )

		# Call Model once to build the model
		_ = model(tf.ones(shape=(1,40)))

		# Take a look at model
		model.summary()
	```
	"""
	def __init__(self, attn_mode, input_shape, input_vocab_size, output_vocab_size, name="Multiplicative-Attention", input_embed_dim=200, hidden_state_dim=200, local_attn_window=None, enc_lstm_dropout=0.2, dec_lstm_dropout=0.2, **kwargs):
		super(multiplicative_attention_model, self).__init__(name=name, **kwargs)

		# Expect attention model to be either global or local
		assert attn_mode == "global" or attn_mode == "local", f"Expect 'global' or 'local' for attn_mode, get {attn_mode} instead"
		self.attn_mode = attn_mode

		# Expect input shape to be two-dimensional
		assert isinstance(input_shape, tuple), "Expect tuple for input_shape, get {type(input_shape)} instead"
		assert len(input_shape) == 2, "Expect 2-dim tuple for input_shape, get {len(input_shape)}-dim instead"
		_, self._n_step = input_shape

		# Embedding layer for input
		self._embedding_layer = Embedding(input_dim=input_vocab_size+1, output_dim=input_embed_dim, input_length=self._n_step)

		# LSTM encoder
		self._lstm_encoder_layer_1 = LSTM(units=hidden_state_dim, activation="tanh", recurrent_activation="sigmoid", use_bias=True, return_sequences=True, return_state=True)
		self._encoder_layer_1_dropout_layer = Dropout(enc_lstm_dropout)
		self._lstm_encoder_layer_2 = LSTM(units=hidden_state_dim, activation="tanh", recurrent_activation="sigmoid", use_bias=True, return_sequences=True, return_state=True)
		self._encoder_layer_2_dropout_layer = Dropout(enc_lstm_dropout)

		# LSTM decoder
		self._lstm_decoder_cell_1 = LSTMCell(units=hidden_state_dim, activation="tanh", recurrent_activation="sigmoid", use_bias=True)
		self._decoder_cell_1_dropout_layer = Dropout(dec_lstm_dropout)
		self._lstm_decoder_cell_2 = LSTMCell(units=hidden_state_dim, activation="tanh", recurrent_activation="sigmoid", use_bias=True)
		self._decoder_dropout_layer = Dropout(dec_lstm_dropout)
		self._decoder_state_array = tf.TensorArray(dtype=tf.float32, size=self._n_step, clear_after_read=True)

		# Attention utility
		if self.attn_mode == "global":
			pass
		if self.attn_mode == "local":
			assert isinstance(local_attn_window, int), f"Expect integer for local_attn_window, get {type(local_attn_window)} instead"
			assert local_attn_window > 0, "Expect positive value for local_attn_window"
			self.D = local_attn_window

		# Utility layer
		self._concat_layer = Concatenate()

		# Output layer
		self._ffnn_decoder_sequence_layer = TimeDistributed(Dense(output_vocab_size+1))
	
	@tf.function
	def call(self, inputs, training=False):
		# Embedding
		inputs_embed = self._embedding_layer(inputs)

		# Encoder
		encoder_1_sequence, encoder_1_hidden_state, encoder_1_cell_state = self._lstm_encoder_layer_1(inputs=inputs_embed, training=training)
		encoder_1_sequence = self._encoder_layer_1_dropout_layer(encoder_1_sequence, training=training)
		encoder_2_sequence, encoder_2_hidden_state, encoder_2_cell_state = self._lstm_encoder_layer_2(inputs=encoder_1_sequence, initial_state=[encoder_1_hidden_state, encoder_1_cell_state], training=training)
		encoder_2_sequence = self._encoder_layer_2_dropout_layer(encoder_2_sequence, training=training)
		decoder_1_input = tf.transpose(encoder_2_sequence, [1, 0, 2])
		decoder_1_states = [encoder_1_hidden_state, encoder_1_cell_state]
		decoder_2_states = [encoder_2_hidden_state, encoder_2_cell_state]

		# Decoder
		for step in range(self._n_step):
			decoder_1_hidden_state, decoder_1_states = self._lstm_decoder_cell_1(inputs=decoder_1_input[step],states=decoder_1_states, training=training)
			decoder_1_hidden_state = self._decoder_cell_1_dropout_layer(decoder_1_hidden_state, training=training)
			decoder_2_hidden_state, decoder_2_states = self._lstm_decoder_cell_2(inputs=decoder_1_hidden_state, states=decoder_2_states, training=training)
			if self.attn_mode == "global":
				attention_score = tf.nn.softmax(tf.einsum("ijk,ik->ij", encoder_2_sequence, decoder_2_hidden_state))
				context_vector = tf.einsum("ijk,ij->ik", encoder_2_sequence, attention_score)
			if self.attn_mode == "local":
				lb, ub = max(0, step-self.D), min(self._n_step-1,step+self.D)
				encoder_2_sequence_part = encoder_2_sequence[:,lb:ub+1,:]
				attention_score = tf.nn.softmax(tf.einsum("ijk,ik->ij", encoder_2_sequence_part, decoder_2_hidden_state))
				context_vector = tf.einsum("ijk,ij->ik", encoder_2_sequence_part, attention_score)
			output_vector = self._concat_layer([decoder_2_hidden_state, context_vector])
			self._decoder_state_array = self._decoder_state_array.write(step,output_vector)
		decoder_sequence = tf.transpose(self._decoder_state_array.stack(), [1, 0, 2])
		decoder_sequence = self._decoder_dropout_layer(decoder_sequence, training=training)

		# Softmax outputs
		outputs = tf.nn.softmax(self._ffnn_decoder_sequence_layer(decoder_sequence))
		return outputs





