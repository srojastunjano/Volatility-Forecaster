import tensorflow as tf
from tensorflow.keras import layers

class EvidentialRegressionHead(layers.Layer):
    def __init__(self, **kwargs):
        super(EvidentialRegressionHead, self).__init__(**kwargs)
        self.dense = layers.Dense(4, activation=None)

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, inputs):

        #pass the latent vector z_tilde through the dense layer
        raw_output = self.dense(inputs)

        gamma_raw, v_raw, alpha_raw, beta_raw = tf.split(raw_output, num_or_size_splits=4, axis=-1)
        eps = 1e-6
        
        gamma = gamma_raw
        
        # Softplus is a smooth approximation of ReLU: log(exp(x) + 1)
        v = tf.nn.softplus(v_raw) + eps
        
        # α (Shape): Softplus + 1 activation. Must be strictly > 1.
        alpha = tf.nn.softplus(alpha_raw) + 1.0 + eps
        
        # β (Scale): Softplus activation. Must be strictly > 0.
        beta = tf.nn.softplus(beta_raw) + eps

        # concatenate the constrained parameters back into a single tensor
        evidential_output = tf.concat([gamma, v, alpha, beta], axis=-1)
        
        return evidential_output