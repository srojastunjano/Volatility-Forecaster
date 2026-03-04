import data_extraction
import tensorflow as tf
from tensorflow.keras import layers
import transformer
import evidential_head
import evidential_loss

def build_ibdl_model(input_shape, d_model=32, num_heads=2, ff_dim=64, num_layers=2, dropout_rate=0.1):
    """
    Combines the Transformer Backbone with the Evidential Head.
    """

    inputs = layers.Input(shape=input_shape)
    
    backbone = transformer.build_transformer_backbone(input_shape, d_model, num_heads, ff_dim, num_layers, dropout_rate)
    z_tilde = backbone(inputs)
    
    # evidential Head to get [gamma, v, alpha, beta]
    evidential_outputs = evidential_head.EvidentialRegressionHead()(z_tilde)
    
    return tf.keras.Model(inputs=inputs, outputs=evidential_outputs, name="IBDL_Volatility_Forecaster")


if __name__ == "__main__":
    # data shape is roughly (samples, 10, 1)
    
    X_train, X_test, y_train, y_test, scaler = data_extraction.prepare_data()

    seq_len = X_train.shape[1]  # 10
    num_features = X_train.shape[2] # 1
    
    model = build_ibdl_model(input_shape=(seq_len, num_features))
    
    # Compile using RMSProp 
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
        loss=evidential_loss.EvidentialLoss(coeff=0.01) # lambda = 0.01
    )
    
    model.summary()
    
    print("\nStarting Model Training (Disentangling Uncertainty)...")
    history = model.fit(
        x=X_train,
        y=y_train,
        batch_size=32,        
        epochs=50,              
        validation_data=(X_test, y_test), 
        verbose=1
    )