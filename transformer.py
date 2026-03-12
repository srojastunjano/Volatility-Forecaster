import tensorflow as tf
from tensorflow.keras import layers
from MCDropout import MCDropout

class PositionalEncoding(layers.Layer):
    def __init__(self, sequence_length, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.pos_encoding = self.positional_encoding(sequence_length, d_model)

    def get_angles(self, pos, i, d_model):
        angles = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return pos * angles
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.pos_encoding.shape[1],
            "d_model": self.pos_encoding.shape[2],
        })
        return config

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model
        )
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.stack([sines, cosines], axis=-1)
        pos_encoding = tf.reshape(pos_encoding, [position, d_model])

        return pos_encoding[tf.newaxis, ...]

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
    

class TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(d_model)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # MC Dropout Layers
        self.dropout1 = MCDropout(dropout_rate)
        self.dropout2 = MCDropout(dropout_rate)

    def call(self, inputs, training=None):
        # Multi-Head Attention + Residual Connection
        attn_output = self.mha(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed Forward + Residual Connection
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config
    

def build_transformer_backbone(input_shape, d_model, num_heads, ff_dim, num_layers, dropout_rate):
    inputs = layers.Input(shape=input_shape) # (None, seq_len, features)
    
    x = layers.Dense(d_model)(inputs)
    x = PositionalEncoding(input_shape[0], d_model)(x)
    
    for _ in range(num_layers):
        x = TransformerEncoderBlock(d_model, num_heads, ff_dim, dropout_rate)(x)
    
    # feature Mapping (Global Pooling to find z-tilde)
    z_tilde = layers.Lambda(lambda x: x[:, -1, :])(x)
    
    return tf.keras.Model(inputs=inputs, outputs=z_tilde, name="Transformer_Backbone")

# Example Initialization:
# (Sequence length = 22, Features = 12 from M-ALL dataset)
backbone = build_transformer_backbone(
    input_shape=(22, 12), 
    d_model=64, 
    num_heads=4, 
    ff_dim=128, 
    num_layers=2, 
    dropout_rate=0.1
)