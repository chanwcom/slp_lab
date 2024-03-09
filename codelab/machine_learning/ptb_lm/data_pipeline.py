from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanw.com@samsung.com)"

import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np

# TODO (chanw.com)
# 1) Updates the comments.
# 2) Adds unit tests for this data pipeline
# 3) Supports None values for dev and test cases.


class TextLineDatasetFactory(object):
    """A factory class for creating text datasets.
    
    Example usage:
        dataset_factory = TextLineDatasetFactory(batch_size=batch_size)

        (train_dataset, dev_dataset, test_dataset) = dataset_factory.create(
            "train_corpus.txt", "dev_corpus.txt", "test_corpus.txt")
    """

    # TODO(chanw.com)
    # We need to have different batch sizes for train, dev, and test sets.
    #
    # TODO(chanw.com)
    # We may need to separate the tokenizer part.
    def __init__(self,
                 add_sos_eos=True,
                 filters="\t\n",
                 oov_token="<unk>",
                 randomize_order=False,
                 batch_size=5,
                 shuffle_buffer_size=256):
        # TODO(chanw.com)
        # Modifies the word_id_example_queue_v2.py
        self._filters = filters
        self._oov_token = oov_token
        self._add_sos_eos = add_sos_eos
        self._randomize_order = randomize_order
        self._shuffle_buffer_size = shuffle_buffer_size
        self._batch_size = batch_size
        self._tokenizer = None

    def create(self, train_text_file_name, dev_text_file_name=None,
               test_text_file_name=None):
        self._tokenizer = self._create_tokenizer(train_text_file_name,
                                                 self._filters,
                                                 self._oov_token)

        train_dataset = self._create_dataset(train_text_file_name,
                                             self._randomize_order,
                                             self._shuffle_buffer_size)

        if dev_text_file_name:
            dev_dataset = self._create_dataset(dev_text_file_name,
                                               self._randomize_order,
                                               self._shuffle_buffer_size)
        else:
            dev_dataset = None

        if test_text_file_name:
            test_dataset = self._create_dataset(test_text_file_name,
                                                self._randomize_order,
                                                self._shuffle_buffer_size)
        else:
            test_dataset = None

        return (train_dataset, dev_dataset, test_dataset)

    @property
    def tokenizer(self):
        return self._tokenizer

    def vocab_size(self):
        assert self._tokenizer, "Tokenizer hasn't been built yet."
        # TODO(chanw.com) Check whether the following is the most efficient
        # way.

        # last_word is the word with the largest index.
        last_word = max(self._tokenizer.word_index,
                        key=self._tokenizer.word_index.get)

        # One is added to include the 0-th index case.
        return self._tokenizer.word_index[last_word] + 1

    def _create_tokenizer(self, corpus_name, filters, oov_token):
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=filters,
                                                          oov_token=oov_token)

        with open(corpus_name, "rt") as in_file:
            lines = in_file.readlines()
            if self._add_sos_eos:
                lines = ["<sos>" + line + "<eos>" for line in lines]

        tokenizer.fit_on_texts(lines)

        return tokenizer

    def _create_dataset(self, corpus_name, randomize_order,
                        shuffle_buffer_size):
        # 1. Creates a tf.data.Dataset object.
        dataset = tf.data.TextLineDataset(corpus_name)

        # 2. Applies shuffling if the randomization option is enabled.
        if randomize_order:
            dataset = dataset.shuffle(shuffle_buffer_size)

        # 3. Skips blank lines by filtering out zero-length utterances.
        dataset = dataset.filter(lambda line: tf.greater(
            tf.strings.length(tf.strings.strip(line)), 0))

        # 4. Converts texts into word IDs.
        dataset = dataset.map(lambda line: tf.py_function(
            self._texts_to_sequence_wrapper, inp=[line], Tout=tf.int32))

        # 5. Filters out zero-length sentences.
        dataset = dataset.filter(lambda line: tf.greater(tf.shape(line)[0], 1))

        # 6. Converts a batch into the (inputs, targets) format.
        #
        # Note that the input format for the Keras "fit" method is
        # (inputs, targets). The input is "line[:-1]" excluding the last <eos>.
        # The target is a sequence containing the next word IDs given by
        # "line[1:]".
        dataset = dataset.map(lambda line: (line[:-1], line[1:]))

        # 7. Performs padded batching.
        dataset = dataset.padded_batch(self._batch_size,
                                       padded_shapes=([None], [None]),
                                       drop_remainder=True)

        return dataset

    def _texts_to_sequence_wrapper(self, line):
        # Needed to used with tf.Dataset.map.
        #
        # Refer to the following link for more details:
        # https://www.tensorflow.org/api_docs/python/tf/data/Dataset?version=stable
        if self._add_sos_eos:
            line = "<sos> " + line + " <eos>"

        return self._tokenizer.texts_to_sequences(
            [line.numpy().decode("utf-8")])
