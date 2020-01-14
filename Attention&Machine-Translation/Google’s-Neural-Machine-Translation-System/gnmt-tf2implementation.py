import tensorflow as tf
from tensorflow.keras.layers import *

assert tf.__version__=="2.0.0", f"Expect TF-2.0.0 but get {tf.__version__}"

class google_neural_machine_translation_model(tf.keras.Model):
	"""A Tensorflow 2.0 implementation of Google Neural Machine Translation model (GNMT) as illustrated in 
	Google’s Neural Machine Translation System.
	
	# Arguments:

		input_shape: shape of input data, should be a two-dimensional tuple (batch, n_step)
		input_vocab_size: size of input vocabulary size excluding <EOS>
		output_vocab_size: size of output vocabulary size excluding <EOS>
		input_embed_dim: word embedding dimension for input
		hidden_state_dim: hidden state dimension for encoder and decoder GRU
		alignment_dim: alignment score's internal dimension
		enc_dropout: fraction of the encoder units to drop
		dec_dropout: fraction of the decoder units to drop
		ffnn_dropout: fraction of the ffnn units to drop
		name: name of the model

	# Paper:

		https://arxiv.org/pdf/1609.08144.pdf%20(7.pdf

	# Examples

	```python
		# Expected shape for input data and output data
		input_shape = (None,40)
		input_vocab_size = 200
		output_vocab_size = 300

		# Initialize Model
		model = google_neural_machine_translation_model(input_shape, input_vocab_size, output_vocab_size)

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
	def __init__(self, input_shape, input_vocab_size, output_vocab_size, name="Google-Neural-Machine-Translation", input_embed_dim=256, hidden_state_dim=256, alignment_dim=128, enc_dropout=0.2, dec_dropout=0.2, ffnn_dropout=0.2, **kwargs):
		super(google_neural_machine_translation_model, self).__init__(name=name, **kwargs)

		# Expect input shape to be two-dimensional
		assert isinstance(input_shape, tuple), "Expect tuple for input_shape, get {type(input_shape)} instead"
		assert len(input_shape) == 2, "Expect 2-dim tuple for input_shape, get {len(input_shape)}-dim instead"
		_, self._n_step = input_shape

		# Expect dropout to be between 0 and 0.3
		assert enc_dropout >= 0 and enc_dropout <= 0.3, f"Expect enc_dropout to between 0 and 0.3, get {enc_dropout} instead"
		self.enc_dropout = enc_dropout
		assert dec_dropout >= 0 and dec_dropout <= 0.3, f"Expect dec_dropout to between 0 and 0.3, get {dec_dropout} instead"
		self.dec_dropout = dec_dropout
		assert ffnn_dropout >= 0 and ffnn_dropout <= 0.3, f"Expect ffnn_dropout to between 0 and 0.3, get {ffnn_dropout} instead"
		self.ffnn_dropout = ffnn_dropout

		# Embedding layer for input
		self._embedding_layer = Embedding(input_dim=input_vocab_size+1, output_dim=input_embed_dim, input_length=self._n_step)

		# Encoder layers
		self._encoder_1_bi_lstm_layer = Bidirectional(LSTM(units=hidden_state_dim, activation="tanh", recurrent_activation="sigmoid", use_bias=True, return_sequences=True, return_state=True, dropout=enc_dropout),  merge_mode="concat")
		self._encoder_2_lstm_layer = LSTM(units=hidden_state_dim, activation="tanh", recurrent_activation="sigmoid", use_bias=True, return_sequences=True, return_state=True, dropout=enc_dropout)
		self._encoder_3_lstm_layer = LSTM(units=hidden_state_dim, activation="tanh", recurrent_activation="sigmoid", use_bias=True, return_sequences=True, return_state=True, dropout=enc_dropout)
		self._encoder_4_lstm_layer = LSTM(units=hidden_state_dim, activation="tanh", recurrent_activation="sigmoid", use_bias=True, return_sequences=True, return_state=True, dropout=enc_dropout)
		self._encoder_5_lstm_layer = LSTM(units=hidden_state_dim, activation="tanh", recurrent_activation="sigmoid", use_bias=True, return_sequences=True, return_state=True, dropout=enc_dropout)
		self._encoder_6_lstm_layer = LSTM(units=hidden_state_dim, activation="tanh", recurrent_activation="sigmoid", use_bias=True, return_sequences=True, return_state=True, dropout=enc_dropout)
		self._encoder_7_lstm_layer = LSTM(units=hidden_state_dim, activation="tanh", recurrent_activation="sigmoid", use_bias=True, return_sequences=True, return_state=True, dropout=enc_dropout)
		self._encoder_8_lstm_layer = LSTM(units=hidden_state_dim, activation="tanh", recurrent_activation="sigmoid", use_bias=True, return_sequences=True, return_state=True, dropout=enc_dropout)

		# Residual connection utility layer
		self._add_layer = Add()

		# Decoder layers
		self._decoder_1_lstm_cell = LSTMCell(units=hidden_state_dim, activation="tanh", recurrent_activation="sigmoid", use_bias=True, dropout=dec_dropout)
		self._decoder_1_sequence_array = tf.TensorArray(dtype=tf.float32, size=self._n_step, clear_after_read=True)
		self._decoder_2_lstm_layer = LSTM(units=hidden_state_dim, activation="tanh", recurrent_activation="sigmoid", use_bias=True, return_sequences=True, return_state=True, dropout=dec_dropout)
		self._decoder_3_lstm_layer = LSTM(units=hidden_state_dim, activation="tanh", recurrent_activation="sigmoid", use_bias=True, return_sequences=True, return_state=True, dropout=dec_dropout)
		self._decoder_4_lstm_layer = LSTM(units=hidden_state_dim, activation="tanh", recurrent_activation="sigmoid", use_bias=True, return_sequences=True, return_state=True, dropout=dec_dropout)
		self._decoder_5_lstm_layer = LSTM(units=hidden_state_dim, activation="tanh", recurrent_activation="sigmoid", use_bias=True, return_sequences=True, return_state=True, dropout=dec_dropout)
		self._decoder_6_lstm_layer = LSTM(units=hidden_state_dim, activation="tanh", recurrent_activation="sigmoid", use_bias=True, return_sequences=True, return_state=True, dropout=dec_dropout)
		self._decoder_7_lstm_layer = LSTM(units=hidden_state_dim, activation="tanh", recurrent_activation="sigmoid", use_bias=True, return_sequences=True, return_state=True, dropout=dec_dropout)
		self._decoder_8_lstm_layer = LSTM(units=hidden_state_dim, activation="tanh", recurrent_activation="sigmoid", use_bias=True, return_sequences=True, return_state=True, dropout=dec_dropout)

		# Attention utility layer 
		self._W_a = self.add_weight(name="W_a", shape=(alignment_dim,hidden_state_dim), dtype=tf.float32)
		self._U_a = self.add_weight(name="U_a", shape=(alignment_dim,hidden_state_dim), dtype=tf.float32)
		self._V_a = self.add_weight(name="V_a", shape=(1,alignment_dim), dtype=tf.float32)
		self._concat_layer = Concatenate()
		self._repeat_layer = RepeatVector(self._n_step)
		self._attention_context_vector_array = tf.TensorArray(dtype=tf.float32, size=self._n_step, clear_after_read=True)

		# Output layer
		self._ffnn_decoder_sequence_layer = TimeDistributed(Dense(output_vocab_size+1))

	@tf.function
	def call(self, inputs, training=False):
		# Embedding
		inputs_embed = self._embedding_layer(inputs)

		# Encoder 1: Bidirectional LSTM without residual connection
		encoder_1_sequences, _, forward_cell_state, backward_hidden_state, backward_cell_state = self._encoder_1_bi_lstm_layer(inputs_embed, training=training)

		# Encoder 2: Unidirectional LSTM without residual connection
		encoder_2_sequences, _, _ = self._encoder_2_lstm_layer(encoder_1_sequences, training=training)

		# Encoder 3: Unidirectional LSTM with residual connection
		encoder_3_sequences_temp, _, _ = self._encoder_3_lstm_layer(encoder_2_sequences, training=training)
		encoder_3_sequences = self._add_layer([encoder_2_sequences, encoder_3_sequences_temp])

		# Encoder 4: Unidirectional LSTM with residual connection
		encoder_4_sequences_temp, _, _ = self._encoder_4_lstm_layer(encoder_3_sequences, training=training)
		encoder_4_sequences = self._add_layer([encoder_3_sequences, encoder_4_sequences_temp])

		# Encoder 5: Unidirectional LSTM with residual connection
		encoder_5_sequences_temp, _, _ = self._encoder_5_lstm_layer(encoder_4_sequences, training=training)
		encoder_5_sequences = self._add_layer([encoder_4_sequences, encoder_5_sequences_temp])

		# Encoder 6: Unidirectional LSTM with residual connection
		encoder_6_sequences_temp, _, _ = self._encoder_6_lstm_layer(encoder_5_sequences, training=training)
		encoder_6_sequences = self._add_layer([encoder_5_sequences, encoder_6_sequences_temp])

		# Encoder 7: Unidirectional LSTM with residual connection
		encoder_7_sequences_temp, _, _ = self._encoder_7_lstm_layer(encoder_6_sequences, training=training)
		encoder_7_sequences = self._add_layer([encoder_6_sequences, encoder_7_sequences_temp])

		# Encoder 8: Unidirectional LSTM with residual connection
		encoder_8_sequences_temp, _, _ = self._encoder_8_lstm_layer(encoder_7_sequences, training=training)
		encoder_8_sequences = self._add_layer([encoder_7_sequences, encoder_8_sequences_temp])

		# To retain as much parallelism as possible during running the decoder layers, 
		# we use the bottom decoder layer output only for obtaining recurrent attention context, 
		# which is sent directly to all the remaining decoder layers. 

		# Decoder 1: Unidirectional LSTM without residual connection, with context vector calculation

		## Initial input for decoder 1
		decoder_1_encoder_input_sequences = tf.transpose(encoder_8_sequences, [1, 0, 2])
		decoder_1_cell_state = backward_cell_state
		decoder_1_hidden_state = backward_hidden_state

		## Execute decoder 1 step by step
		for step in range(self._n_step):
			## Obtain context vectors
			W_a_dot_decoder = self._repeat_layer(tf.einsum("ij,kj->ik", decoder_1_hidden_state, self._W_a))
			U_a_dot_encoder = tf.einsum("ijk,zk->ijz", encoder_8_sequences, self._U_a)
			V_a_dot_tanh = tf.einsum("ijk,zk->ij", tf.nn.tanh(self._add_layer([W_a_dot_decoder,U_a_dot_encoder])), self._V_a)
			alignment_score = tf.nn.softmax(V_a_dot_tanh)
			context_vector = tf.einsum("ij,ijz->iz", alignment_score, encoder_8_sequences)

			## Store context vectors
			self._attention_context_vector_array = self._attention_context_vector_array.write(step, context_vector)

			## Decoder 1 call
			decoder_1_input = self._concat_layer([decoder_1_encoder_input_sequences[step], context_vector])
			_, decoder_1_states = self._decoder_1_lstm_cell(inputs=decoder_1_input, states=[decoder_1_hidden_state, decoder_1_cell_state], training=training)
			decoder_1_hidden_state, decoder_1_cell_state = decoder_1_states[0], decoder_1_states[1]

			## Store decoder 1 hidden state
			self._decoder_1_sequence_array = self._decoder_1_sequence_array.write(step, decoder_1_hidden_state)

		## Transpose decoder 1 hidden states from (step, batch, feature) to (batch, step, feature)
		decoder_1_sequences = tf.transpose(self._decoder_1_sequence_array.stack(), [1, 0, 2])

		## Transpose context vectors from (step, batch, feature) to (batch, step, feature)
		context_vectors = tf.transpose(self._attention_context_vector_array.stack(), [1, 0, 2])

		## Reset decoder 1 dropout mask
		self._decoder_1_lstm_cell.reset_dropout_mask()

		# Decoder 2: Unidirectional LSTM without residual connection, context vector acquired from decoder 1
		decoder_2_input_sequences = self._concat_layer([decoder_1_sequences, context_vectors])
		decoder_2_sequences, _, _ = self._decoder_2_lstm_layer(decoder_2_input_sequences, training=training)

		# Decoder 3: Unidirectional LSTM with residual connection, context vector acquired from decoder 1
		decoder_3_input_sequences = self._concat_layer([decoder_2_sequences, context_vectors])
		decoder_3_sequences_temp, _, _ = self._decoder_3_lstm_layer(decoder_3_input_sequences, training=training)
		decoder_3_sequences = self._add_layer([decoder_2_sequences, decoder_3_sequences_temp])

		# Decoder 4: Unidirectional LSTM with residual connection, context vector acquired from decoder 1
		decoder_4_input_sequences = self._concat_layer([decoder_3_sequences, context_vectors])
		decoder_4_sequences_temp, _, _ = self._decoder_4_lstm_layer(decoder_4_input_sequences, training=training)
		decoder_4_sequences = self._add_layer([decoder_3_sequences, decoder_4_sequences_temp])

		# Decoder 5: Unidirectional LSTM with residual connection, context vector acquired from decoder 1
		decoder_5_input_sequences = self._concat_layer([decoder_4_sequences, context_vectors])
		decoder_5_sequences_temp, _, _ = self._decoder_5_lstm_layer(decoder_5_input_sequences, training=training)
		decoder_5_sequences = self._add_layer([decoder_4_sequences, decoder_5_sequences_temp])

		# Decoder 6: Unidirectional LSTM with residual connection, context vector acquired from decoder 1
		decoder_6_input_sequences = self._concat_layer([decoder_5_sequences, context_vectors])
		decoder_6_sequences_temp, _, _ = self._decoder_6_lstm_layer(decoder_6_input_sequences, training=training)
		decoder_6_sequences = self._add_layer([decoder_5_sequences, decoder_6_sequences_temp])

		# Decoder 7: Unidirectional LSTM with residual connection, context vector acquired from decoder 1
		decoder_7_input_sequences = self._concat_layer([decoder_6_sequences, context_vectors])
		decoder_7_sequences_temp, _, _ = self._decoder_7_lstm_layer(decoder_7_input_sequences, training=training)
		decoder_7_sequences = self._add_layer([decoder_6_sequences, decoder_7_sequences_temp])

		# Decoder 8: Unidirectional LSTM with residual connection, context vector acquired from decoder 1
		decoder_8_input_sequences = self._concat_layer([decoder_7_sequences, context_vectors])
		decoder_8_sequences_temp, _, _ = self._decoder_8_lstm_layer(decoder_8_input_sequences, training=training)
		decoder_8_sequences = self._add_layer([decoder_7_sequences, decoder_8_sequences_temp])

		# Softmax Outputs
		if training: decoder_8_sequences = tf.nn.dropout(decoder_8_sequences, self.ffnn_dropout)
		outputs = tf.nn.softmax(self._ffnn_decoder_sequence_layer(decoder_8_sequences))
		return outputs
	