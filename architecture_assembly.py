import data_extraction
import tensorflow as tf
from tensorflow.keras import layers
import transformer
import evidential_head
import evidential_loss

def build_ibdl_model(input_shape, d_model=32, num_heads=2, ff_dim=64, num_layers=2, dropout_rate=0.2):
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
    
    X_train, X_test, y_train, y_test, scaler = data_extraction.prepare_data('NVDA','max', 22)

    seq_len = X_train.shape[1] 
    num_features = X_train.shape[2] #
    
    model = build_ibdl_model(input_shape=(seq_len, num_features))
    
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001), # high drop out rate leads to unstability in NIG
        loss=evidential_loss.EvidentialLoss(coeff=0.001) 
    )
    
    model.summary()

    early_stopper = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',    
        patience=10,           
        restore_best_weights=True, # reverts the model to the best epoch
        verbose=1
    )
    
    print("\nStarting Model Training (Disentangling Uncertainty)...")
    history = model.fit(
        x=X_train,
        y=y_train,
        batch_size=32,        
        epochs=80,              
        validation_data=(X_test, y_test), 
        callbacks=[early_stopper],
        verbose=1
    )

    model.save("ibdl_volatility_1y_v5.keras")