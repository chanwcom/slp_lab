from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanw.com@samsung.com)"

# Third-party imports
import tensorflow as tf
from packaging import version

assert version.parse(tf.__version__) > version.parse("2.0.0"), (
    "At least tensorflow 2.0 is required.")

def create_lm_model(batch_size, unit_size, target_size, dropout_rate):
    """Creates a Keras Language Model (LM) using the functional API.

    Args:
        batch_size: The input batch size. Note that the input batch size
            is needed for the stateful LSTM layers. Refer to the following
            page about the reason.
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN
        unit_size: The LSTM cell size.
        target_size: The number of target classes.
        dropout_rate: The input dropout rate used in this model.

    Returns:
        The created Keras LM model.
    """
    inputs = tf.keras.layers.Input(batch_shape=(batch_size, None))
    embedding = tf.keras.layers.Embedding(target_size,
                                          unit_size,
                                          batch_input_shape=(batch_size,
                                                             None))(inputs)
    dropout0 = tf.keras.layers.Dropout(dropout_rate,
                                       noise_shape=(batch_size, 1,
                                                    unit_size))(embedding)
    lstm0 = tf.keras.layers.LSTM(unit_size,
                                 return_sequences=True,
                                 stateful=True)(dropout0)
    dropout1 = tf.keras.layers.Dropout(dropout_rate,
                                       noise_shape=(batch_size, 1,
                                                    unit_size))(lstm0)
    lstm1 = tf.keras.layers.LSTM(unit_size,
                                 return_sequences=True,
                                 stateful=True)(dropout1)
    dropout2 = tf.keras.layers.Dropout(dropout_rate,
                                       noise_shape=(batch_size, 1,
                                                    unit_size))(lstm1)
    softmax = tf.keras.layers.Dense(target_size,
                                    activation="softmax")(dropout2)

    return tf.keras.models.Model(inputs=inputs, outputs=softmax)
