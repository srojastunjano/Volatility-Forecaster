import tensorflow as tf
from tensorflow.keras import layers

class EvidentialRegressionHead(layers.Layer):
    def __init__(self, name="evidential_head", **kwargs):
        super(EvidentialRegressionHead, self).__init__(name=name, **kwargs)
        # Note: Layers will automatically use the global mixed_bfloat16 policy
        self.dense_gamma = layers.Dense(1, activation=None, name="dense_gamma")
        self.dense_uncertainty = layers.Dense(3, activation=None, name="dense_uncertainty")

    def call(self, inputs):
        # Cast inputs to float32 immediately to avoid bf16 rounding errors in Softplus
        z = tf.cast(inputs, dtype=tf.float32)
        
        gamma = self.dense_gamma(z)
        uncertainty = self.dense_uncertainty(z)

        v_raw, alpha_raw, beta_raw = tf.split(uncertainty, num_or_size_splits=3, axis=-1)
        eps = 1e-7
        
        # Perform all transcendental math in float32
        v = tf.nn.softplus(v_raw) + eps
        alpha = tf.nn.softplus(alpha_raw) + 1.0 + eps
        beta = tf.nn.softplus(beta_raw) + eps

        return tf.concat([gamma, v, alpha, beta], axis=-1)
