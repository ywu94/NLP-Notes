import tensorflow as tf
from tensorflow.keras.layers import *

assert tf.__version__=="2.0.0", f"Expect TF-2.0.0 but get {tf.__version__}"

class additive_attention_model(tf.keras.Model):
	"""A Tensorflow 2.0 implementation of encoder-decoder attention model as illustrated in 
	Neural Machine Translation by Jointly Learning to Align and Translate.
	
	# Arguments:

		input_shape: shape of input data, should be a two-dimensional tuple (batch, n_step)
		input_vocab_size: size of input vocabulary size excluding <EOS>
		output_vocab_size: size of output vocabulary size excluding <EOS>
		input_embed_dim: word embedding dimension for input
		hidden_state_dim: hidden state dimension for encoder and decoder GRU
		alignment_dim: alignment score's internal dimension
		enc_gru_dropout: fraction of the encoder gru units to drop
		dec_gru_dropout: fraction of the decoder gru units to drop
		name: name of the model

	# Paper:

		https://arxiv.org/pdf/1409.0473.pdf

	# Examples

	```python
		# Expected shape for input data and output data
		input_shape = (None,40)
		input_vocab_size = 200
		output_vocab_size = 300

		# Initialize Model
		model = additive_attention_model(input_shape, input_vocab_size, output_vocab_size)

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
	def __init__(self, input_shape, input_vocab_size, output_vocab_size, name="Additive-Attention", input_embed_dim=200, hidden_state_dim=200, alignment_dim=40, enc_gru_dropout=0.2, dec_gru_dropout=0.2, **kwargs):
		super(additive_attention_model, self).__init__(name=name, **kwargs)

		# Expect input shape to be two-dimensional
		assert isinstance(input_shape, tuple), "Expect tuple for input_shape, get {type(input_shape)} instead"
		assert len(input_shape) == 2, "Expect 2-dim tuple for input_shape, get {len(input_shape)}-dim instead"
		_, self._n_step = input_shape

		# Embedding layer for input
		self._embedding_layer = Embedding(input_dim=input_vocab_size+1, output_dim=input_embed_dim, input_length=self._n_step)

		# Bidirectional GRU encoder
		self._bi_gru_encoder_layer = Bidirectional(GRU(units=hidden_state_dim, activation="tanh", recurrent_activation="sigmoid", use_bias=True, return_sequences=True, return_state=True), merge_mode="concat")
		self._encoder_dropout_layer = Dropout(enc_gru_dropout)

		# Weights and utility layers for calculating alignment score
		self._W_a = self.add_weight(name="W_a", shape=(alignment_dim,hidden_state_dim), dtype=tf.float32)
		self._U_a = self.add_weight(name="U_a", shape=(alignment_dim,hidden_state_dim*2), dtype=tf.float32)
		self._V_a = self.add_weight(name="V_a", shape=(1,alignment_dim), dtype=tf.float32)
		self._repeat_layer = RepeatVector(self._n_step)
		self._additive_layer = Add()
		self._concat_layer = Concatenate()

		# GRU decoder
		self._decoder_cell = GRUCell(units=hidden_state_dim, activation="tanh", recurrent_activation="sigmoid", use_bias=True)
		self._decoder_dropout_layer = Dropout(dec_gru_dropout)
		self._decoder_state_array = tf.TensorArray(dtype=tf.float32, size=self._n_step, clear_after_read=True)
		self._ffnn_decoder_sequence_layer = TimeDistributed(Dense(output_vocab_size+1))
	
	@tf.function	
	def call(self, inputs, training=False):
		# Embedding
		inputs_embed = self._embedding_layer(inputs)

		# Encoder
		encoder_sequence, forward_state, backward_state = self._bi_gru_encoder_layer(inputs_embed, training=training)
		encoder_sequence = self._encoder_dropout_layer(encoder_sequence, training=training)
		decoder_hidden_state = backward_state
		encoder_sequence_transpose = tf.transpose(encoder_sequence, [1, 0, 2])

		# Decoder
		for step in range(self._n_step):
			W_a_dot_decoder_state_sequence = self._repeat_layer(tf.einsum("ij,kj->ik", decoder_hidden_state, self._W_a))
			U_a_dot_encoder_state_sequence = tf.einsum("ijk,zk->ijz", encoder_sequence, self._U_a)
			V_a_dot_tanh = tf.einsum("ijk,zk->ij", tf.nn.tanh(self._additive_layer([W_a_dot_decoder_state_sequence,U_a_dot_encoder_state_sequence])), self._V_a)
			alignment_score = tf.nn.softmax(V_a_dot_tanh)
			attention_vector = tf.einsum("ij,ijz->iz", alignment_score, encoder_sequence)
			decoder_input = self._concat_layer([encoder_sequence_transpose[step], attention_vector])
			decoder_hidden_state, _ = self._decoder_cell(inputs=decoder_input, states=[decoder_hidden_state], training=training)
			self._decoder_state_array = self._decoder_state_array.write(step, decoder_hidden_state)
		decoder_sequence = tf.transpose(self._decoder_state_array.stack(), [1, 0, 2])
		decoder_sequence = self._decoder_dropout_layer(decoder_sequence, training=training)

		# Softmax outputs
		outputs = tf.nn.softmax(self._ffnn_decoder_sequence_layer(decoder_sequence))
		return outputs
