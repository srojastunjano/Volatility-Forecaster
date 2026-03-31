import numpy as np
import tensorflow as tf
import os


# transforms npy files into tfrecord files for NVIDIA optimisation
def create_tf_example(temporal_row, contextual_row, target_val):
    feature = {
        'temporal': tf.train.Feature(float_list=tf.train.FloatList(value=temporal_row.flatten())),
        'contextual': tf.train.Feature(float_list=tf.train.FloatList(value=contextual_row.flatten())),
        'target': tf.train.Feature(float_list=tf.train.FloatList(value=[target_val]))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def convert_to_tfrecords_split(npz_path, train_path, val_path, split_ratio=0.8):
    print(f"Loading data from {npz_path} via mmap...")
    # mmap_mode='r' ensures we don't load the massive array into RAM
    data = np.load(npz_path, mmap_mode='r')
    temporal = data['temporal']
    contextual = data['contextual']
    target = data['target']

    total_samples = len(target)
    split_idx = int(total_samples * split_ratio)

    print(f"\nTotal samples found: {total_samples}")
    print(f"Chronological Split Index: {split_idx}")
    print(f"Train: {split_ratio*100:.0f}% | Validation: {(1-split_ratio)*100:.0f}%\n")

    print(f"--- Writing Training Records to {train_path} ---")
    with tf.io.TFRecordWriter(train_path) as writer:
        for i in range(split_idx):
            example = create_tf_example(temporal[i], contextual[i], target[i])
            writer.write(example.SerializeToString())
            
            if i > 0 and i % 250000 == 0:
                print(f"  -> Serialized {i} training samples...")

    print(f"\n--- Writing Validation Records to {val_path} ---")
    with tf.io.TFRecordWriter(val_path) as writer:
        for i in range(split_idx, total_samples):
            example = create_tf_example(temporal[i], contextual[i], target[i])
            writer.write(example.SerializeToString())
            
            samples_written = i - split_idx
            if samples_written > 0 and samples_written % 100000 == 0:
                print(f"  -> Serialized {samples_written} validation samples...")

    print("\n TFRecord conversion complete. Data is ready for streaming.")

if __name__ == "__main__":
    convert_to_tfrecords_split(
        npz_path='global_panel_data.npz',
        train_path='global_panel_train.tfrecord',
        val_path='global_panel_val.tfrecord',
        split_ratio=0.8
    )
