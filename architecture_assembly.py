import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
from tensorflow.keras import mixed_precision
import os
import numpy as np
import tensorflow as tf


# NVDIA GDX SPARK BLACKWELL OPTIMIZATION
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=-1'
# # os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=-1'
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# # os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit --tf_xla_enable_xla_devices' 
# # Also try limiting the PTX version explicitly if the above doesn't work:
# os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# # tf.config.optimizer.set_jit(False)
# tf.config.run_functions_eagerly(False)
# tf.keras.backend.set_floatx('float32')

# policy = tf.keras.mixed_precision.Policy('mixed_bfloat16')
# tf.keras.mixed_precision.set_global_policy(policy)

# #gpus = tf.config.list_physical_devices('GPU')
# #if gpus:
#    # try:
#      #   for gpu in gpus:
#     #        tf.config.experimental.set_memory_growth(gpu, True)
#    # except RuntimeError as e:
#    #     print(e)
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         # On DGX Spark, we sometimes need to set a hard limit to prevent system-wide crashes
#         tf.config.set_logical_device_configuration(
#             gpus[0],
#             [tf.config.LogicalDeviceConfiguration(memory_limit=102400)] # 100GB
#         )
#     except RuntimeError as e:
#         print(f"Memory config error: {e}")

from tensorflow.keras import layers
from tensorflow.keras.mixed_precision import Policy, set_global_policy
from transformer import build_transformer_backbone
from evidential_head import EvidentialRegressionHead
from evidential_loss import EvidentialLoss

def extract_panel_data():
    data = np.load('global_panel_data.npz')
    X_temporal_global = data['temporal']   
    X_contextual_global = data['contextual']
    y_global = data['target']
    return X_temporal_global, X_contextual_global, y_global

def build_generator(file_path, start_idx, end_idx):
    def generator():
        data = np.load(file_path, mmap_mode='r')
        for i in range(start_idx, end_idx):
            yield (
                {
                    "temporal_input": data['temporal'][i], 
                    "contextual_input": data['contextual'][i]
                },
                data['target'][i]
            )
    return generator

def parse_fn(example_proto):
    feature_description = {
        'temporal': tf.io.FixedLenFeature([21, 1], tf.float32),
        'contextual': tf.io.FixedLenFeature([2], tf.float32),
        'target': tf.io.FixedLenFeature([], tf.float32),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    return {"temporal_input": parsed['temporal'], "contextual_input": parsed['contextual']}, parsed['target']

def create_fast_dataset(tfrecord_path, batch_size):
    ds = tf.data.TFRecordDataset(tfrecord_path, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(10000).batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def create_streaming_dataset(file_path, start_idx, end_idx, batch_size):
    gen = build_generator(file_path, start_idx, end_idx)
    
    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            {
                "temporal_input": tf.TensorSpec(shape=(21, 1), dtype=tf.float32),
                "contextual_input": tf.TensorSpec(shape=(2,), dtype=tf.float32)
            },
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    )
    ds = ds.shuffle(buffer_size=60) 
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def create_in_memory_dataset(X_temp, X_cont, y, batch_size):
    # Loads everything into memory as a highly optimized TF Dataset
    ds = tf.data.Dataset.from_tensor_slices((
        {"temporal_input": X_temp, "contextual_input": X_cont}, 
        y
    ))
    # Cache, shuffle, batch, and prefetch!
    # ds = ds.cache() # Locks data in RAM
    ds = ds.shuffle(buffer_size=10000) # True global shuffle
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE) # Feeds GPU asynchronously
    return ds



# MODEL ARCHITECTURE

def build_ibdl_model(temporal_shape, contextual_shape, d_model=128, num_heads=2, ff_dim=256, num_layers=4, dropout_rate=0.2):
    temporal_input = layers.Input(shape=temporal_shape, name="temporal_input")
    contextual_input = layers.Input(shape=contextual_shape, name="contextual_input")
    
    z_temporal = build_transformer_backbone(
        input_shape=temporal_shape, 
        d_model=d_model, 
        num_heads=num_heads, 
        ff_dim=ff_dim, 
        num_layers=num_layers, 
        dropout_rate=dropout_rate
    )(temporal_input)
    
    z_contextual = layers.Dense(64, activation="relu", name="contextual_dense")(contextual_input)
    merged_vectors = layers.Concatenate(name="fusion_concat")([z_temporal, z_contextual])
    z_fused = layers.Dense(128, activation="relu", name="mixing_chamber_dense")(merged_vectors)
    evidential_outputs = EvidentialRegressionHead(name="evidential_head")(z_fused)
    
    return tf.keras.Model(
        inputs=[temporal_input, contextual_input], 
        outputs=evidential_outputs, 
        name="Global_Hybrid_IBDL_Forecaster"
    )


# MAIN EXECUTION
if __name__ == "__main__":
    print("\n Generating S&P 500 Global Hybrid Panel ")
    npz_file_path = 'global_panel_data.npz'
    
    with np.load(npz_file_path) as temp_data:
        total_samples = len(temp_data['target'])
        split_idx = int(total_samples * 0.8)
    
    print(f"\nTotal Samples Found: {total_samples}")
    print(f"Split Index: {split_idx}")

    X_temporal_global, X_contextual_global, y_global = extract_panel_data()
    
    split_idx = int(len(X_temporal_global) * 0.8)
    
    X_temp_train = X_temporal_global[:split_idx]
    X_temp_val = X_temporal_global[split_idx:]
    
    X_cont_train = X_contextual_global[:split_idx]
    X_cont_val = X_contextual_global[split_idx:]
    
    y_train = y_global[:split_idx]
    y_val = y_global[split_idx:]

    # Instantiate the streaming datasets
    train_ds = create_in_memory_dataset(X_temp_train, X_cont_train, y_train, batch_size=8192) # GPU needs to work so CPU has time to prepare the next batch... consider 16384
    test_ds = create_in_memory_dataset(X_temp_val, X_cont_val, y_val, batch_size=8192) # TO do: change this one as well and look for the variabels

    # train_ds = create_streaming_dataset(npz_file_path, 0, split_idx, batch_size=16384)
    # test_ds = create_streaming_dataset(npz_file_path, split_idx, total_samples, batch_size=16384)
    
    #train_ds = create_fast_dataset('global_panel_train.tfrecord', batch_size=16384)
    #test_ds = create_fast_dataset('global_panel_val.tfrecord', batch_size=16384)
    print("\n--- Chronological Split Complete (Streaming Active) ---")


    temp_shape = (21, 1)
    cont_shape = (2,)
    model = build_ibdl_model(temporal_shape=temp_shape, contextual_shape=cont_shape)
    model.summary()

    print("\n--- PHASE 1: Global Warmup (Freezing Uncertainty) ---")
    ev_head = model.get_layer("evidential_head") 
    ev_head.dense_uncertainty.trainable = False 
    
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001), 
        loss=EvidentialLoss(coeff=0.01),
    )
    
    early_stopper_phase1 = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
    )
    
    model.fit(
    	train_ds,
        epochs=40,           
        validation_data=test_ds, 
        callbacks=[early_stopper_phase1],
        verbose=1
    )

    print("\n--- PHASE 2: Calibrated Unfreezing (Learning the Physics of Risk) ---")
    ev_head.dense_uncertainty.trainable = True
    
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.00001), 
        loss=EvidentialLoss(coeff=0.1),
    )
    
    early_stopper_phase2 = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=40, restore_best_weights=True, verbose=1
    )
    
    history = model.fit(
        train_ds,
        epochs=300,              
        validation_data=test_ds, 
        callbacks=[early_stopper_phase2],
        verbose=1
    )

    model.save("global_hybrid_ibdl_v1.keras")
    print("\n Training Complete. Global Hybrid Model saved.")
   
   
