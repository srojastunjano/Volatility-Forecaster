import tensorflow as tf
from tensorflow.keras import layers

class MCDropout(layers.Dropout):
    def __init__(self, rate, **kwargs):
        super(MCDropout, self).__init__(rate, **kwargs)

    def call(self, inputs, training=None):
        """
        Overrides the standard call method.
        By explicitly passing training=True to the parent class, 
        dropout remains active during model.predict(), enabling 
        the generation of the Credal Set.
        """
        return super().call(inputs, training=True)