import tensorflow as tf
from tensorflow.keras.layers import *

assert tf.__version__>="2.0.0", f"Expect TF>=2.0.0 but get {tf.__version__}"

from .tf2_util_layer import TransformerEncoder, TransformerDecoder

class transformer_model(tf.keras.Model):
	"""A Tensorflow 2.0 implementation of transformer model as illustrated in 
	Attention is What You Need.
	
	# Arguments:

		input_vocab_size: size of input vocabulary size excluding <EOS>
		target_vocab_size: size of target vocabulary size excluding <EOS>
		enc_n_layer: number of encoder layer to stack
		dec_n_layer: number of decoder layer to stack
		attn_n_head: number of heads to use in multi-head attention
		d_model: last dimension of attention vector
		d_ff: inner dimension for point wise feed forward network
		dropout_rate: drop out rate

	# Paper:

		https://arxiv.org/abs/1706.03762

	# Examples

	```python
		# Sample Data
		## Note that as described in the original paper, the second dimension of inputs and targets should also be the same.
		inputs = tf.zeros(shape=(100,10))
		targets = tf.zeros(shape=(100,10))

		# Hyperparameter
		n_enc_layer = 6
		n_dec_layer = 6
		d_model = 512
		attn_n_head = 8
		d_ff = 2048
		input_vocab_size = 100
		target_vocab_size = 100

		# Initialize Model
		model = transformer_model(n_enc_layer, n_dec_layer, d_model, attn_n_head, d_ff, input_vocab_size, target_vocab_size)

		# Compile Model
		learning_rate = 1e-3
		model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
					  optimizer=tf.keras.optimizers.Adam(learning_rate),
					  metrics=[tf.keras.metrics.sparse_categorical_accuracy]
					  )

		# Call Model once to build the model
		enc_padding_mask, dec_comb_mask, dec_padding_mask = model.get_masks(inputs, targets)
		_ = model(inputs, targets, enc_padding_mask=enc_padding_mask, dec_comb_mask=dec_comb_mask, dec_padding_mask=dec_padding_mask)

		# Take a look at model
		model.summary()
	```
	"""
	def __init__(self, n_enc_layer, n_dec_layer, d_model, attn_n_head, d_ff, input_vocab_size, target_vocab_size, dropout_rate=0.1, **kwargs):
		super(transformer_model, self).__init__(**kwargs)

		self._encoder = TransformerEncoder(input_vocab_size=input_vocab_size, n_layer=n_enc_layer, d_model=d_model, attn_n_head=attn_n_head, d_ff=d_ff, dropout_rate=dropout_rate)
		self._decoder = TransformerDecoder(target_vocab_size=target_vocab_size, n_layer=n_dec_layer, d_model=d_model, attn_n_head=attn_n_head, d_ff=d_ff, dropout_rate=dropout_rate)
		self._ffnn_layer = Dense(target_vocab_size)

	@tf.function
	def call(self, inputs, targets, training=None, enc_padding_mask=None, dec_comb_mask=None, dec_padding_mask=None):
		enc_outputs = self._encoder(inputs, mask=enc_padding_mask, training=training)
		dec_outputs = self._decoder(targets, enc_outputs, comb_mask=dec_comb_mask, padding_mask=dec_padding_mask, training=training)
		ffnn_outputs = self._ffnn_layer(dec_outputs)

		return tf.nn.softmax(ffnn_outputs)

	def get_padding_mask(self, inputs):
		"""
		Mask Zero Padding Mask.
		"""
		inputs_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32) # (batch_size, n_step)
		return inputs_mask

	def get_forward_mask(self, dim_0, dim_1):
		"""
		Look ahead mask to maintain autoregression.
		"""
		forward_mask = 1 - tf.linalg.band_part(tf.ones((dim_0, dim_1)), -1, 0) # (dim_0, dim_1)
		return forward_mask

	def get_masks(self, inputs, targets):
		# Create padding mask for encoder's multi-head attention layer
		enc_padding_mask = self.get_padding_mask(inputs) # (batch_size, n_step)
		enc_padding_mask = tf.expand_dims(enc_padding_mask, 1) # (batch_size, 1, n_step/n_key)

		# Create padding mask for decoder's second multi-head attention layer which uses encoder outputs for Query and Key Tensors.
		dec_padding_mask = self.get_padding_mask(inputs) # (batch_size, n_step)
		dec_padding_mask = tf.expand_dims(dec_padding_mask, 1) # (batch_size, 1, n_step/n_key)

		# Create mask for decoder's first multi-head attention layer, which uses decoder inputs for Query, Key, and Value Tensors.
		## Create look ahead mask 
		dec_comb_forward_mask = self.get_forward_mask(tf.shape(targets)[-1], tf.shape(targets)[-1]) # (n_query, n_key)
		dec_comb_forward_mask = tf.expand_dims(dec_comb_forward_mask, 0) # (1, n_query, n_key)
		dec_comb_padding_mask = self.get_padding_mask(targets) # (batch_size, n_step)
		dec_comb_padding_mask = tf.expand_dims(dec_comb_padding_mask, 1) # (batch_size, 1, n_step/n_key)
		dec_comb_mask = dec_comb_forward_mask + dec_comb_padding_mask # (batch_size, n_query, n_key)

		return enc_padding_mask, dec_comb_mask, dec_padding_mask
























