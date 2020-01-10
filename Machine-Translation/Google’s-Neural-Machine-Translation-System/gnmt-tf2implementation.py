import tensorflow as tf
from tensorflow.keras.layers import *

assert tf.__version__=="2.0.0", f"Expect TF-2.0.0 but get {tf.__version__}"

class google_neural_machine_translation_model(tf.keras.Model):
	def __init__(self, input_shape, input_vocab_size, output_vocab_size, name="Google-Neural-Machine-Translation", input_embed_dim=200, hidden_state_dim=200):
		super(google_neural_machine_translation_modelm self).__init__(name=name, **kwargs)

	def call(self, inputs, training=False):
		pass
	