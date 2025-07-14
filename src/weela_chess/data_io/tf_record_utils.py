from pathlib import Path

import tensorflow as tf
import numpy as np

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
    """Serializes a NumPy array into a string."""
    return tf.io.serialize_tensor(array)

def array_example(array):
    """Creates a tf.train.Example message from a NumPy array."""
    feature = {
        'array_raw': _bytes_feature(serialize_array(array)),
        # 'shape': _int64_feature(np.prod(array.shape)),
        # 'dtype': _int64_feature(array.dtype.num)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def save_array_as_tfrecord(array: np.ndarray, filename: Path):
    """Saves a NumPy array to a TFRecord file."""
    with tf.io.TFRecordWriter(str(filename.absolute())) as writer:
        example = array_example(array)
        writer.write(example.SerializeToString())

def parse_tfrecord_meta(example_proto):
    feature_description = {
        'array_raw': tf.io.FixedLenFeature([], tf.string),
        # 'shape': tf.io.FixedLenFeature([], tf.int64),
        # 'dtype': tf.io.FixedLenFeature([], tf.int64),
    }
    features = tf.io.parse_single_example(example_proto, feature_description)
    return features

def parse_tfrecord_full(example_proto, dtype):
    feature_description = {
        'array_raw': tf.io.FixedLenFeature([], tf.string),
        # 'shape': tf.io.FixedLenFeature([], tf.int64),
        # 'dtype': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    array_feature = example["array_raw"]
    tensor = tf.io.parse_tensor(array_feature, out_type=dtype)
    # tensor = tf.reshape(tensor, shape)
    return tensor
