import tensorflow as tf
from tensorflow.keras.layers import *

assert tf.__version__=="2.0.0", f"Expect TF-2.0.0 but get {tf.__version__}"

class transformer_model(tf.keras.Model):
	def __init__(self):
		pass

	def call(self, inputs, training=False):
		pass