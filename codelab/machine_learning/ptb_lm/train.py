#!/usr/bin/python3
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Required version: Tensorflow 2.0.0-beta1
# TODO(chanw.com) Implement a version checking routine.

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

import numpy as np
import tensorflow as tf
import os
from tensorflow import keras

from data_pipeline import TextLineDatasetFactory
from model import create_lm_model

# Generates reproducible results in this tutorial code.
np.random.seed(0)
tf.compat.v1.set_random_seed(0)

# TODO(chanw.com) This code gives only the approximate measure of perplexity.
# If we use the class, then we may resolve the problem.
def perplexity_metrics(y_true, y_pred):
    """
    https://stackoverflow.com/questions/41881308/how-to-calculate-perplexity-of-rnn-in-tensorflow
    https://github.com/keras-team/keras/issues/8267
    """

    # For some unknown reason, for the first function call, the dimension of
    # y_true is [None, None, None].  Thus, we force the dimension of y_true.
    # May be a bug in Tensorflow/Keras?
    y_true = tf.cast(tf.reshape(y_true, (y_pred.get_shape().as_list()[0], -1)),
                     tf.int32)

    # First, obtains the mean of sequence examples in a batch for each
    # sample index. After doing this, summation is done.
    cross_entropy = tf.math.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_true, logits=tf.math.log(y_pred)))

    return tf.math.exp(cross_entropy)


class MyCustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, model=None, dev_dataset=None):
        super(MyCustomCallback, self).__init__()
        self._model = model
        self._dev_dataset = dev_dataset

    def on_epoch_end(self, epoch, logs=None):
        print("{0}-th epoch has finished.".format(epoch))


# Initialization parameters.
batch_size = 20
cell_size = 200
dropout_rate = 0.0
init_learning_rate = 0.1


dataset_factory = TextLineDatasetFactory(batch_size=batch_size)
(train_dataset, dev_dataset,
 test_dataset) = dataset_factory.create("./data/ptb_train.txt",
                                        "./data/ptb_valid.txt",
                                        "./data/ptb_test.txt")
vocab_size = dataset_factory.vocab_size()

lm_model = create_lm_model(batch_size, cell_size, vocab_size, dropout_rate)
lm_model.compile(loss="sparse_categorical_crossentropy",
                 optimizer=tf.keras.optimizers.SGD(learning_rate=init_learning_rate),
                 metrics=["accuracy", perplexity_metrics])

lm_model.summary()

checkpoint_dir = './training_checkpoints'

# The Name of the checkpoint files.
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

lm_model.fit(train_dataset,
             epochs=100,
             callbacks=[
                 tf.keras.callbacks.TensorBoard(log_dir='./logs'),
                 tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                                    save_weights_only=True),
                 MyCustomCallback(lm_model)
             ])

lm_model.evaluate(test_dataset)
