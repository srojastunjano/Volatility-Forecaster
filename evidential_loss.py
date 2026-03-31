import tensorflow as tf
import numpy as np

class EvidentialLoss(tf.keras.losses.Loss):
    def __init__(self, coeff=0.05, name="evidential_loss", **kwargs): 
        """
        Args:
            coeff (float): The lambda (λ) hyperparameter. Controls the weight 
                           of the regularization term.
        """
        super(EvidentialLoss, self).__init__(name=name, **kwargs)
        self.coeff = coeff

    def call(self, y_true, y_pred):
        # Ensure predictions and targets are float32
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        y_true = tf.cast(tf.reshape(y_true, (-1, 1)), dtype=tf.float32)

        gamma, v, alpha, beta = tf.split(y_pred, 4, axis=-1)
        omega = 2.0 * beta * (1.0 + v)
        
        # Stable NLL using log-gamma
        nll = (0.5 * tf.math.log(np.pi / v)
               - alpha * tf.math.log(omega)
               + (alpha + 0.5) * tf.math.log(v * tf.square(y_true - gamma) + omega)
               + tf.math.lgamma(alpha)
               - tf.math.lgamma(alpha + 0.5))

        reg = tf.abs(y_true - gamma) * (2.0 * v + alpha)
        
        return tf.reduce_mean(nll + (self.coeff * reg))
