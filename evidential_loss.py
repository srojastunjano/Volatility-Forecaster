import tensorflow as tf
import numpy as np

class EvidentialLoss(tf.keras.losses.Loss):
    def __init__(self, coeff=1e-2, name="evidential_loss", **kwargs):
        """
        Args:
            coeff (float): The lambda (λ) hyperparameter. Controls the weight 
                           of the regularization term. Higher values produce a 
                           more "humble" model with wider uncertainty bounds.
        """
        super(EvidentialLoss, self).__init__(name=name, **kwargs)
        self.coeff = coeff

    def call(self, y_true, y_pred):
        gamma, v, alpha, beta = tf.split(y_pred, num_or_size_splits=4, axis=-1)
        
        y_true = tf.cast(y_true, dtype=gamma.dtype)
        y_true = tf.reshape(y_true, (-1, 1))

        # scaling factor based on beta and evidence
        omega = 2.0 * beta * (1.0 + v)
        
        # NLL calculation using log-gamma function for mathematical stability
        nll = (
            0.5 * tf.math.log(np.pi / v)
            - alpha * tf.math.log(omega)
            + (alpha + 0.5) * tf.math.log(v * tf.square(y_true - gamma) + omega)
            + tf.math.lgamma(alpha)
            - tf.math.lgamma(alpha + 0.5)
        )

        # Evidential Regularizer (Humility)

        # Penalizes high evidence when the absolute error is high
        error = tf.abs(y_true - gamma)
        evidence = 2.0 * v + alpha
        reg_loss = error * evidence
        total_loss = nll + (self.coeff * reg_loss)
        
        return tf.reduce_mean(total_loss)