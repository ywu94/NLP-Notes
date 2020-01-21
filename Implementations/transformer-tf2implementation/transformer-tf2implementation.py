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
		# Expected shape for input data and output data
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
		_ = model(tf.zeros(shape=(1,40)), tf.zeros(shape=(1,40)))

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
	def call(self, inputs, targets, training=None):
		enc_outputs = self._encoder(inputs)
		dec_outputs = self._decoder(targets, enc_outputs)
		ffnn_outputs = self._ffnn_layer(dec_outputs)

		return tf.nn.softmax(ffnn_outputs)

