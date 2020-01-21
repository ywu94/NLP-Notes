import tensorflow as tf
from tensorflow.keras.layers import *

assert tf.__version__>="2.0.0", f"Expect TF>=2.0.0 but get {tf.__version__}"

class PositionalSinEmbedding(tf.keras.layers.Layer):
	"""
	Positional Sinusoidal Embedding layer  as described in "Attention is All You Need".
	|
	| Parameters:
	| | input_dim: parameter for embedding layer
	| | output_dim: parameter for embedding layer
	| | mask_zero: parameter for embedding layer
	|
	| Inputs: two dimensional tensor to be embedded
	|
	| Outputs: 
	| | inputs_embed: tensor of shape (batch_size, n_step, output_dim)
	| | inputs_positional_encoding: tensor of shape (n_step, output_dim)
	"""
	def __init__(self, input_dim, output_dim, mask_zero=True, **kwargs):
		super(PositionalSinEmbedding, self).__init__(**kwargs)
		self._input_dim = input_dim
		self._output_dim = output_dim
		self._mask_zero = mask_zero
		self._embedding_layer = Embedding(input_dim=input_dim, output_dim=output_dim, mask_zero=mask_zero)

	@tf.function
	def call(self, inputs, training=False, mask=None):
		inputs_embed = self._embedding_layer(inputs)
		inputs_positional_encoding = self.get_positional_encoding(tf.shape(inputs_embed)[-2], tf.shape(inputs_embed)[-1])
		return inputs_embed, inputs_positional_encoding

	def get_positional_encoding(self, n_step, n_embed):
		"""
		Formula:
		| PE(pos,2i) = sin(pos/10000**(2*i/d_model))
		| PE(pos,2i+1) = cos(pos/10000**(2*i/d_model))
		Return:
		| Tensor of shape (n_step, n_embed)
		"""
		n_step = tf.cast(n_step, tf.float32)
		n_embed = tf.cast(n_embed,tf.float32)

		step_dim = tf.cast(tf.reshape(tf.range(n_step), shape=(-1,1)), tf.float32)
		embed_dim = tf.cast(tf.range(n_embed), tf.float32)
		positional_encoding = step_dim * (1.0/tf.math.pow(10000.0, (2.0*(embed_dim//2.0))/n_embed))
		positional_encoding = tf.where(tf.cast(tf.range(n_embed)%2.0, tf.bool), tf.math.cos(positional_encoding), tf.math.sin(positional_encoding))
		return positional_encoding

	def compute_padding_mask(self, inputs, mask=None):
		"""
		Compute mask
		"""
		if self._mask_zero: 
			bool_mask = self._embedding_layer.compute_mask(inputs, mask=mask)
			return tf.where(bool_mask, tf.zeros(shape=tf.shape(bool_mask)), tf.ones(shape=tf.shape(bool_mask)))
		return None

	def get_config(self):
		"""
		Get configuration for current layer
		"""
		config = super(PositionalSinEmbedding, self).get_config()
		cur_config = {"input_dim":self._input_dim, "output_dim":self._output_dim, "mask_zero":self._mask_zero}
		config.update(cur_config)
		return config

class MultiHeadAttention(tf.keras.layers.Layer):
	"""
	Multi-Head Attention layer as described in "Attention is All You Need".
	|
	| Parameters:
	| | n_head: number of heads to use
	| | d_model: last dimension of attention vector
	|
	| Inputs:
	| | Q: query tensor of shape (batch_size, n_query, _)
	| | K: key tensor of shape (batch_size, n_key, _)
	| | V: value tensor of shape (batch_size, n_key, _)
	|
	| Outputs:
	| | attn_vector: vector of shape (batch_size, n_query, d_model)
	| 
	| Notes:
	| | All assertations in call function are depreciated as it's not well supported in Autograph yet.
	"""
	def __init__(self, n_head=8, d_model=512, **kwargs):
		super(MultiHeadAttention, self).__init__(**kwargs)

		# Parameter setting: d_k = d_v = d_model/h, in original paper, d_model=512, h=8, d_k=d_v=64
		# The dimension of each head is reduced to reduce computational cost
		assert d_model%n_head==0, f"Illegal parameter n_head:{n_head}, d_model:{d_model}"
		self._n_head = n_head
		self._d_model = d_model
		self._d_k = d_model//n_head
		self._d_v = d_model//n_head

		self._Wq_list = [Dense(self._d_k) for _ in range(self._n_head)]
		self._Wk_list = [Dense(self._d_k) for _ in range(self._n_head)]
		self._Wv_list = [Dense(self._d_v) for _ in range(self._n_head)]

		self._concat_layer = Concatenate()
		self._Wo = Dense(self._d_model)

	@tf.function
	def call(self, Q, K, V, training=None, forward_mask=None, padding_mask=None):
		"""
		Input:
		| Q: Tensor of shape (batch_size, n_query, _)
		| K: Tensor of shape (batch_size, n_key, _)
		| V: Tensor of shape (batch_size, n_key, _)
		| forward_mask: whether to use forward mask
		| padding_mask: Tensor of shape (batch_size, n_query)
		"""
		# assert tf.shape(K)[-2] == tf.shape(V)[-2], f"K has shape {tf.shape(K)} while V has shape {tf.shape(V)}"

		Q_list = [Wq(Q) for Wq in self._Wq_list] # list of (batch_size, n_query, d_k)
		K_list = [Wk(K) for Wk in self._Wk_list] # list of (batch_size, n_key, d_k)
		V_list = [Wv(V) for Wv in self._Wv_list] # list of (batch_size, n_key, d_v)

		forward_mask = self.get_forward_mask(tf.shape(Q)[-2], tf.shape(K)[-2]) if forward_mask is not None else None # (1, n_query, n_key)
		padding_mask = self.get_padding_mask(padding_mask) if padding_mask is not None else None # (batch_size, 1, n_key)

		if forward_mask is not None:
			attn_list = [self.get_scaled_dot_product_attention(q, k, v, mask=forward_mask) for q, k, v in zip(Q_list, K_list, V_list)] # list of (batch_size, n_query, d_v)
		elif padding_mask is not None:
			attn_list = [self.get_scaled_dot_product_attention(q, k, v, mask=padding_mask) for q, k, v in zip(Q_list, K_list, V_list)] # list of (batch_size, n_query, d_v)
		else:
			attn_list = [self.get_scaled_dot_product_attention(q, k, v, mask=None) for q, k, v in zip(Q_list, K_list, V_list)] # list of (batch_size, n_query, d_v)
		concated_attn = self._concat_layer(attn_list) # (batch_size, n_query, n_head * d_v)
		multihead_attn = self._Wo(concated_attn) # (batch_size, n_query, d_model)

		return multihead_attn

	def get_padding_mask(self, padding_mask):
		"""
		Reshape padding mask from (batch_size, n_key) to (batch_size, 1, n_key)
		"""
		return tf.reshape(padding_mask, (tf.shape(padding_mask)[0], 1, tf.shape(padding_mask)[1]))

	def get_forward_mask(self, dim_0, dim_1):
		"""
		Intution:
		| Mask future token of current sequences
		Inplementation:
		| Use a matrix with only upper triangular part being 1
		Output:
		| Tensor of shape (1, dim_0, dim_1)
		"""
		return tf.reshape(1 - tf.linalg.band_part(tf.ones((dim_0, dim_1)), -1, 0), (1, dim_0, dim_1))

	def get_scaled_dot_product_attention(self, Q, K, V, mask=None):
		"""
		Input:
		| Q: Tensor of shape (batch_size, n_query, d_k)
		| K: Tensor of shape (batch_size, n_key, d_k)
		| V: Tensor of shape (batch_size, n_key, d_v)
		| mask: Tensor of shape 
		| | (n_query, n_key) 
		Formula:
		| attn = softmax(Q.dot(K.T)/sqrt(d_k))V
		Output:
		| attn_vector: Tensor of shape (batch_size, n_query, d_v)
		"""
		# assert tf.shape(Q)[-1] == tf.shape(K)[-1], f"Q has shape {tf.shape(Q)} while K has shape {tf.shape(K)}"
		# assert tf.shape(K)[-2] == tf.shape(V)[-2], f"K has shape {tf.shape(K)} while V has shape {tf.shape(V)}"

		QK = tf.matmul(Q, K, transpose_b=True) # (batch_size, n_query, n_key)
		QK_scale = QK/tf.math.sqrt(tf.cast(tf.shape(Q)[-1], tf.float32)) # (batch_size, n_query, n_key)

		if mask is not None:
			QK_scale += (mask * -1e9) # (batch_size, n_query, n_key)

		attn_weight = tf.nn.softmax(QK_scale, axis=-1) # (batch_size, n_query, n_key)
		attn_vector = tf.matmul(attn_weight, V) # (batch_size, n_query, d_v)

		return attn_vector

	def get_config(self):
		"""
		Get configuration for current layer
		"""
		config = super(MultiHeadAttention, self).get_config()
		cur_config = {"n_head":self._n_head, "d_model":self._d_model}
		config.update(cur_config)
		return config

class PointWiseFeedForwardNetwork(tf.keras.layers.Layer):
	"""
	Point wise feed forward network as described in "Attention is All You Need".
	|
	| Parameters:
	| | d_model: output dimension
	| | d_ff: middle dimension
	"""
	def __init__(self, d_model=512, d_ff=2048, **kwargs):
		super(PointWiseFeedForwardNetwork, self).__init__(**kwargs)
		self._d_model = d_model
		self._d_ff = d_ff
		self._dense_layer_1 = Dense(d_ff, activation="relu")
		self._dense_layer_2 = Dense(d_model)

	@tf.function
	def call(self, inputs, training=None):
		outputs = self._dense_layer_1(inputs)
		outputs = self._dense_layer_2(outputs)
		return outputs

	def get_config(self):
		"""
		Get configuration for current layer
		"""
		config = super(PointWiseFeedForwardNetwork, self).get_config()
		cur_config = {"d_model":self._d_model, "d_ff":self._d_ff}
		config.update(cur_config)
		return config

class TransformerEncoderLayer(tf.keras.layers.Layer):
	"""
	Encoder layer as described in "Attention is All You Need".
	|
	| Parameters:
	| | attn_n_head: number of heads to use
	| | d_model: last dimension of attention vector
	| | d_ff: inner dimension for point wise feed forward network
	| | dropout_rate: drop out rate
	|
	| Inputs:
	| | Inputs: query tensor of shape (batch_size, n_step, d_model)
	|
	| Outputs:
	| | Outputs: vector of shape (batch_size, n_step, d_model)
	|
	| Notes:
	| | 1. As described in the original paper, to use residual connection the input and output data should share the same dimension.
	| | 2. Multi-head attention in encoder layer only uses padding mask.
	"""
	def __init__(self, attn_n_head, d_model, d_ff, dropout_rate=0.1, **kwargs):
		super(TransformerEncoderLayer, self).__init__(**kwargs)

		self._attn_n_head = attn_n_head
		self._d_model = d_model
		self._d_ff = d_ff
		self._dropout_rate = dropout_rate

		self._multi_head_attn_layer = MultiHeadAttention(attn_n_head, d_model)
		self._layer_norm_1 = LayerNormalization(epsilon=1e-6)
		self._dropout_layer_1 = Dropout(dropout_rate)
		self._ffnn_layer = PointWiseFeedForwardNetwork(d_model=d_model, d_ff=d_ff)
		self._layer_norm_2 = LayerNormalization(epsilon=1e-6)
		self._dropout_layer_2 = Dropout(dropout_rate)

	@tf.function
	def call(self, inputs, training=None, forward_mask=None, padding_mask=None):
		mh_attn = self._multi_head_attn_layer(inputs, inputs, inputs, forward_mask=forward_mask, padding_mask=padding_mask)
		mh_attn = self._dropout_layer_1(mh_attn, training=training)
		ffnn_input = self._layer_norm_1(mh_attn+inputs)

		ffnn_output = self._ffnn_layer(ffnn_input)
		ffnn_output = self._dropout_layer_2(ffnn_output, training=training)
		outputs = self._layer_norm_2(ffnn_input+ffnn_output)

		return outputs

	def get_config(self):
		"""
		Get configuration for current layer
		"""
		config = super(TransformerEncoderLayer, self).get_config()
		cur_config = {"attn_n_head":self._attn_n_head, "d_model":self._d_model, "d_ff":self._d_ff, "dropout_rate":self._dropout_rate}
		config.update(cur_config)
		return config

class TransformerEncoder(tf.keras.layers.Layer):
	"""
	Encoder as described in "Attention is All You Need".
	|
	| Parameters:
	| | input_vocab_size: parameter for embedding layer
	| | n_layer: number of encoder layer to stack
	| | attn_n_head: number of heads to use
	| | d_model: last dimension of attention vector
	| | d_ff: inner dimension for point wise feed forward network
	| | dropout_rate: drop out rate
	|
	| Inputs:
	| | Inputs: query tensor of shape (batch_size, n_step, d_model)
	|
	| Outputs:
	| | Outputs: vector of shape (batch_size, n_step, d_model)
	|
	| Notes:
	| | 1. As described in the original paper, to use residual connection the input and output data should share the same dimension.
	| | 2. Multi-head attention in encoder layer only uses padding mask.
	"""
	def __init__(self, input_vocab_size, n_layer, d_model, attn_n_head, d_ff, dropout_rate=0.1, **kwargs):
		super(TransformerEncoder, self).__init__(**kwargs)

		self._input_vocab_size = input_vocab_size
		self._n_layer = n_layer
		self._d_model = d_model
		self._attn_n_head = attn_n_head
		self._d_ff = d_ff,
		self._dropout_rate = dropout_rate

		self._positional_embedding_layer = PositionalSinEmbedding(input_vocab_size, d_model)
		self._enc_layer_list = [TransformerEncoderLayer(attn_n_head, d_model, d_ff, dropout_rate=dropout_rate) for _ in range(n_layer)]
		self._dropout_layer = Dropout(dropout_rate)

	@tf.function
	def call(self, inputs, training=None):
		inputs_embed, inputs_positional_encoding = self._positional_embedding_layer(inputs)
		# Make up for the scaled attention
		inputs_embed *= tf.math.sqrt(tf.cast(self._d_model, tf.float32))
		inputs_pos_embed = inputs_embed + inputs_positional_encoding
		inputs_padding_mask = self._positional_embedding_layer.compute_padding_mask(inputs)

		outputs = self._dropout_layer(inputs_pos_embed, training=training)

		for i in range(self._n_layer):
			outputs = self._enc_layer_list[i](outputs, training=training, padding_mask=inputs_padding_mask)

		return outputs

	def get_config(self):
		"""
		Get configuration for current layer
		"""
		config = super(TransformerEncoder, self).get_config()
		cur_config = {"input_vocab_size":self._input_vocab_size, "n_layer":self._n_layer, "attn_n_head":self._attn_n_head, "d_model":self._d_model, "d_ff":self._d_ff, "dropout_rate":self._dropout_rate}
		config.update(cur_config)
		return config

class TransformerDecoderLayer(tf.keras.layers.Layer):
	"""
	Encoder layer as described in "Attention is All You Need".
	|
	| Parameters:
	| | attn_n_head: number of heads to use
	| | d_model: last dimension of attention vector
	| | d_ff: inner dimension for point wise feed forward network
	| | dropout_rate: drop out rate
	|
	| Inputs:
	| | Inputs: query tensor of shape (batch_size, n_step, d_model)
	|
	| Outputs:
	| | Outputs: vector of shape (batch_size, n_step, d_model)
	|
	| Notes:
	| | As described in the original paper, to use residual connection the input and output data should share the same dimension.
	"""
	def __init__(self, attn_n_head, d_model, d_ff, dropout_rate=0.1, **kwargs):
		super(TransformerDecoderLayer, self).__init__(**kwargs)

		self._attn_n_head = attn_n_head
		self._d_model = d_model
		self._d_ff = d_ff
		self._dropout_rate = dropout_rate

		self._multi_head_attn_layer_1 = MultiHeadAttention(attn_n_head, d_model)
		self._multi_head_attn_layer_2 = MultiHeadAttention(attn_n_head, d_model)
		self._layer_norm_1 = LayerNormalization(epsilon=1e-6)
		self._layer_norm_2 = LayerNormalization(epsilon=1e-6)
		self._layer_norm_3 = LayerNormalization(epsilon=1e-6)
		self._dropout_layer_1 = Dropout(dropout_rate)
		self._dropout_layer_2 = Dropout(dropout_rate)
		self._dropout_layer_3 = Dropout(dropout_rate)
		self._point_wise_feed_forward_network_layer = PointWiseFeedForwardNetwork(d_model, d_ff)

	def call(self, inputs, enc_inputs, training=None, forward_mask=True, padding_mask=None):
		mh_attn_1 = self._multi_head_attn_layer_1(inputs, inputs, inputs, forward_mask=forward_mask)
		mh_attn_1 = self._dropout_layer_1(mh_attn_1, training=training)
		inputs_1 = self._layer_norm_1(mh_attn_1 + inputs)

		mh_attn_2 = self._multi_head_attn_layer_2(enc_inputs, enc_inputs, inputs_1, padding_mask=padding_mask)
		mh_attn_2 = self._dropout_layer_2(mh_attn_2, training=training)
		inputs_2 = self._layer_norm_2(mh_attn_2 + inputs_1)

		ffnn_output = self._point_wise_feed_forward_network_layer(inputs_2)
		ffnn_output = self._dropout_layer_3(ffnn_output, training=training)
		outputs = self._layer_norm_3(inputs_2 + ffnn_output)

		return outputs

	def get_config(self):
		"""
		Get configuration for current layer
		"""
		config = super(TransformerDecoderLayer, self).get_config()
		cur_config = {"attn_n_head":self._attn_n_head, "d_model":self._d_model, "d_ff":self._d_ff, "dropout_rate":self._dropout_rate}
		config.update(cur_config)
		return config

class TransformerDecoder(tf.keras.layers.Layer):
	"""
	Decoder as described in "Attention is All You Need".
	|
	| Parameters:
	| | target_vocab_size: parameter for embedding layer
	| | n_layer: number of decoder layer to stack
	| | attn_n_head: number of heads to use
	| | d_model: last dimension of attention vector
	| | d_ff: inner dimension for point wise feed forward network
	| | dropout_rate: drop out rate
	|
	| Inputs:
	| | inputs: tensor of shape (batch_size, n_step, d_model)
	| | enc_inputs: tensor of shape (batch_size, n_step, d_model)
	|
	| Outputs:
	| | outputs: vector of shape (batch_size, n_step, d_model)
	|
	| Notes:
	| | 1. As described in the original paper, to use residual connection the input and output data should share the same dimension.
	| | 2. Multi-head attention in encoder layer only uses padding mask.
	"""	
	def __init__(self, target_vocab_size, n_layer, d_model, attn_n_head, d_ff, dropout_rate=0.1, **kwargs):
		super(TransformerDecoder, self).__init__(**kwargs)

		self._target_vocab_size = target_vocab_size
		self._n_layer = n_layer
		self._d_model = d_model
		self._attn_n_head = attn_n_head
		self._d_ff = d_ff,
		self._dropout_rate = dropout_rate

		self._positional_embedding_layer = PositionalSinEmbedding(target_vocab_size, d_model)
		self._dec_layer_list = [TransformerDecoderLayer(attn_n_head, d_model, d_ff, dropout_rate=dropout_rate) for _ in range(n_layer)]
		self._dropout_layer = Dropout(dropout_rate)

	@tf.function
	def call(self, inputs, enc_inputs, training=None):
		inputs_embed, inputs_positional_encoding = self._positional_embedding_layer(inputs)
		# Make up for the scaled attention
		inputs_embed *= tf.math.sqrt(tf.cast(self._d_model, tf.float32))
		inputs_pos_embed = inputs_embed + inputs_positional_encoding
		inputs_padding_mask = self._positional_embedding_layer.compute_padding_mask(inputs)

		outputs = self._dropout_layer(inputs_pos_embed, training=training)

		for i in range(self._n_layer):
			outputs = self._dec_layer_list[i](outputs, enc_inputs, training=training, padding_mask=inputs_padding_mask)

		return outputs

	def get_config(self):
		"""
		Get configuration for current layer
		"""
		config = super(TransformerDecoder, self).get_config()
		cur_config = {"target_vocab_size":self._target_vocab_size, "n_layer":self._n_layer, "attn_n_head":self._attn_n_head, "d_model":self._d_model, "d_ff":self._d_ff, "dropout_rate":self._dropout_rate}
		config.update(cur_config)
		return config





