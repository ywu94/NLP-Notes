import time

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
		# Note that as described in the original paper, the second dimension of inputs and targets should also be the same.
		inputs = tf.zeros(shape=(100,10))
		targets = tf.zeros(shape=(100,10))

		# Optional codes for ensuring that the second dimension of inputs and targets being the same.
		# Note that this part of codes is not supported in tf.function as we use python boolean value here.
		# This can be addressed by rewritting using tensorflow built-in functions.
		step_diff = tf.shape(inputs)[-1] - tf.shape(targets)[-1]
		# Pad if inputs has longer sequences
		if step_diff > 0:
			targets = tf.pad(targets,[[0,0],[0,step_diff]])
		# Trim if inputs has shorter sequences
		elif step_diff < 0:
			targets = targets[:,:step_diff]

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

		# Call Model once to build the model
		_ = model(inputs, targets)

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
		enc_padding_mask, dec_comb_mask, dec_padding_mask = model.get_masks(inputs, targets)
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

	def get_forward_mask(self, dimension):
		"""
		Look ahead mask to maintain autoregression.
		"""
		forward_mask = 1 - tf.linalg.band_part(tf.ones((dimension, dimension)), -1, 0) # (dimension, dimension)
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
		dec_comb_forward_mask = self.get_forward_mask(tf.shape(inputs)[-1]) # (n_query, n_key)
		dec_comb_forward_mask = tf.expand_dims(dec_comb_forward_mask, 0) # (1, n_query, n_key)
		dec_comb_padding_mask = self.get_padding_mask(targets) # (batch_size, n_step)
		dec_comb_padding_mask = tf.expand_dims(dec_comb_padding_mask, 1) # (batch_size, 1, n_step/n_key)
		dec_comb_mask = dec_comb_forward_mask + dec_comb_padding_mask # (batch_size, n_query, n_key)

		return enc_padding_mask, dec_comb_mask, dec_padding_mask

# Training Script from https://www.tensorflow.org/tutorials/text/transformer

# # Model Optimizer
# class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
# 	def __init__(self, d_model, warmup_steps=4000):
# 		super(CustomSchedule, self).__init__()

# 		self.d_model = d_model
# 		self.d_model = tf.cast(self.d_model, tf.float32)
# 		self.warmup_steps = warmup_steps
# 	def __call__(self, step):
# 		arg1 = tf.math.rsqrt(step)
# 		arg2 = step * (self.warmup_steps ** -1.5)
# 		return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

# learning_rate = CustomSchedule(d_model)
# optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# # Loss Function
# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

# def loss_function(real, pred):
# 	mask = tf.math.logical_not(tf.math.equal(real, 0))
# 	loss = loss_object(real, pred)
# 	mask = tf.cast(mask, dtype=loss_.dtype)
# 	loss *= mask
# 	return tf.reduce_mean(loss)

# # Model Training Checkpoint
# checkpoint_path = "./checkpoints/train"
# ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
# ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# # If a checkpoint exists, restore the latest checkpoint.
# if ckpt_manager.latest_checkpoint:
# 	ckpt.restore(ckpt_manager.latest_checkpoint)
# 	print ('Latest checkpoint restored!!')

# train_step_signature = [
# 	tf.TensorSpec(shape=(None, None), dtype=tf.int64),
# 	tf.TensorSpec(shape=(None, None), dtype=tf.int64),
# ]

# @tf.function(input_signature=train_step_signature)
# def train_step(inp, tar):
# 	tar_inp = tar[:,:-1]
# 	tar_real = tar[:,1:]

# 	with tf.GradientTape() as tape:
# 	predictions = transformer(inp, tar_inp)
# 	loss = loss_function(tar_real, predictions)

# 	gradients = tape.gradient(loss, transformer.trainable_variables)    
# 	optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

# 	train_loss(loss)
# 	train_accuracy(tar_real, predictions)

# EPOCHS = 20

# for epoch in range(EPOCHS):
# 	start = time.time()
# 	train_loss.reset_states()
# 	train_accuracy.reset_states()

# 	# inp -> portuguese, tar -> english
# 	for (batch, (inp, tar)) in enumerate(train_dataset):
# 		train_step(inp, tar)

# 		if batch % 50 == 0:
# 		print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, batch, train_loss.result(), train_accuracy.result()))

# 	if (epoch + 1) % 5 == 0:
# 		ckpt_save_path = ckpt_manager.save()
# 		print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,ckpt_save_path))

# 	print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))
# 	print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
