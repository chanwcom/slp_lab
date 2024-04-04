"""A module implementing the subsampling layer.

The following classes are implemented.
 * Conv1DSubsampling
"""

# pylint: disable=no-member, import-error

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanw.com@samsung.com)"

# Standard imports
import abc
import enum
import math

# Third-party imports
import numpy as np
import tensorflow as tf
from packaging import version

# Custom imports
# TODO(chanw.com) Think about moving the location of the following util module.
from math_lib.operation import util
from machine_learning.layers import dropout
from machine_learning.layers import layer_params_pb2
from machine_learning.layers import layer_type
from speech.trainer.ck_trainer.util import proto_util
from speech.trainer.tf_based_end_to_end_trainer.tf_trainer.layers import normalization

assert version.parse(tf.__version__) >= version.parse("2.0.0"), (
    "At least tensorflow 2.0 is required.")

LayerType = layer_type.Type


class SubsamplingFactory(object):
    """A factory class to create a concrete Subsampling layer."""
    def __init__(self) -> None:
        # Creates a dict containing all the classes derived from Subsampling.
        #
        # Note that Subsampling is an "Abstract" class defined in the same
        # module below.
        self._layer_dict = util.create_sub_cls_dict(Subsampling)

    def create(self, params_proto):
        assert isinstance(params_proto, layer_params_pb2.SubsamplingParams), (
            "The type of \"params_proto\" should be SubsamplingParams.")

        DEFAULT_CLASS_NAME = "Conv1DSubsampling"
        class_name = proto_util.get_field(params_proto, "class_name",
                                          DEFAULT_CLASS_NAME)

        return self._layer_dict[class_name](params_proto)


def _is_power_of_two(value: int) -> bool:
    """Checks whether an input value is a power of two.

    This implementation is based on the suggestion shown in the following page:
    https://stackoverflow.com/questions/57025836/how-to-check-if-a-given-number-is-a-power-of-two.
    Returns:
        True if the input value is a power of two, otherwise it is False.
    """
    return (value & (value - 1) == 0) and value != 0


class Subsampling(tf.keras.layers.Layer, abc.ABC):
    """An abstract class for Subsampling."""
    @abc.abstractmethod
    def __init__(self,
                 params_proto: layer_params_pb2.SubsamplingParams) -> None:
        """Initializes a Subsampling object."""
        super(Subsampling, self).__init__()

        assert isinstance(params_proto, layer_params_pb2.SubsamplingParams), (
            "The type of \"params_proto\" should be SubsamplingParams.")

        self._subsampling_factor = proto_util.get_field(
            params_proto, "subsampling_factor", 4)

    @property
    def subsampling_factor(self) -> int:
        return self._subsampling_factor


class Conv1DSubsampling(Subsampling):
    """A Keras layer implementation of Subsampling using Conv1D."""

    # TODO(chanw.com) Check whether Conv2D is better than Conv1D.
    # TODO(chanw.com) Finds the optimal Kernel size.
    def __init__(self,
                 params_proto: layer_params_pb2.SubsamplingParams) -> None:
        super(Conv1DSubsampling, self).__init__(params_proto)
        assert params_proto.class_name == self.__class__.__name__, (
            f"\"class_name\" must be {self.__class__.__name__}.")

        try:
            conv_params_proto = layer_params_pb2.Conv1DSubsamplingParams()
            params_proto.class_params.Unpack(conv_params_proto)
        except:
            raise ValueError(
                "The \"class_params\" field must be an any params proto packing "
                "an object of the type Conv1DSubsamplingParams.")

        assert _is_power_of_two(self.subsampling_factor), (
            "The \"subsampling_factor\" must be a power of two.")

        DEFAULT_NUM_FILTERANK_CHANNELS = 40
        DEFAULT_NUM_CONV_CHANNELS = 256
        DEFAULT_KERNEL_SIZE = 5

        self._num_filterbank_channels = proto_util.get_field(
            conv_params_proto, "num_filterbank_channels",
            DEFAULT_NUM_FILTERANK_CHANNELS)
        assert self._num_filterbank_channels > 0

        num_conv_channels = proto_util.get_field(conv_params_proto,
                                                 "num_conv_channels",
                                                 DEFAULT_NUM_CONV_CHANNELS)
        assert num_conv_channels > 0

        self._kernel_size = proto_util.get_field(conv_params_proto,
                                                 "kernel_size",
                                                 DEFAULT_KERNEL_SIZE)
        num_conv1d_layers = int(np.log2(self.subsampling_factor))

        self._layers = []
        self._layer_types = []

        # TODO(chanw.com) Support different types of normalizations such as
        # GroupNormalization, LayerNormalization, and so on.
        self._layers.append(normalization.BatchNormWithMask(sync=True))
        self._layer_types.append(LayerType.BATCH_NORM)

        if conv_params_proto.HasField("dropout_params"):
            factory = dropout.DropoutFactory()

        for _ in range(num_conv1d_layers):
            if conv_params_proto.HasField("dropout_params"):
                self._layers.append(
                    factory.create(conv_params_proto.dropout_params))
                self._layer_types.append(LayerType.DROPOUT)

            self._layers.append(
                tf.keras.layers.Conv1D(filters=num_conv_channels,
                                       kernel_size=self._kernel_size,
                                       strides=2,
                                       padding="same"))
            self._layer_types.append(LayerType.CONV1D)

        self._layers.append(
            tf.keras.layers.Dense(units=self._num_filterbank_channels))
        self._layer_types.append(LayerType.DENSE)

    # yapf: disable
    def call(self, inputs: dict, training: bool=None) -> dict:
        # yapf: enable
        """Applies this Keras layer to a batch of input sequences.

        Args:
            inputs: A dictionary containing a batch of inputs and lengths.
                 The keys are as follows:
                "SEQ_DATA": A tensor containing a batch of input sequences.
                    The shape is (batch_size, sequence_len, feature_size).
                "SEQ_LEN": The length of sequence inputs.
                    The shape is (batch_size, 1).

            training: A flag which is "True" when called for training.

        Returns:
            A dictionary containing a batch of outputs and lengths.
                The keys are as follows:
                "SEQ_DATA": A tensor containing a batch of output sequences.
                    The shape is (batch_size,
                    int(sequence_len / subsampling_factor), feature_size).
                "SEQ_LEN": The length of sequence inputs.
                    The shape is (batch_size, 1).
        """
        assert (isinstance(inputs, dict)
                and {"SEQ_DATA", "SEQ_LEN"} <= inputs.keys()), (
                    "The inputs must be a dictionary containing \"SEQ_DATA\""
                    " and \"SEQ_LEN\" as keys.")
        assert len(inputs["SEQ_DATA"].shape) == 3, (
            "The rank of inputs[\"SEQ_DATA\"] must be three.")

        tf.debugging.assert_equal(
            tf.shape(inputs["SEQ_DATA"])[2], self._num_filterbank_channels,
            "The input to this layer does not have the expected number of "
            "channels.")

        inputs_seq = inputs["SEQ_DATA"]
        seq_len = inputs["SEQ_LEN"]

        for (layer, layer_type) in zip(self._layers, self._layer_types):
            if layer_type == LayerType.BATCH_NORM:
                mask = tf.cast(tf.sequence_mask(
                    seq_len, maxlen=tf.shape(inputs_seq)[1]),
                               dtype=inputs_seq.dtype)
                # Provides the mask if the layer is "BatchNormWithMask".
                outputs = layer(inputs_seq, mask=mask, training=training)
            else:
                outputs = layer(inputs_seq, training=training)

            if layer_type == LayerType.CONV1D:
                # In obtaining the sequence length, "ceil" is used rather than
                # "floor", since the padding type in MaxPool1D is "same" rather
                # than the default "valid". Refer to the following page:
                #
                # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D
                seq_len = tf.math.ceil(tf.math.divide(seq_len, 2))
                mask = tf.expand_dims(tf.cast(tf.sequence_mask(
                    seq_len, maxlen=tf.shape(outputs)[1]),
                                              dtype=inputs_seq.dtype),
                                      axis=2)
                outputs = tf.math.multiply(outputs, mask)

            inputs_seq = outputs

        # Now, the dimension of inputs_seq is as follows:
        # (batch_size, M / reduction_factor, num_channels).
        # We changes the dimension of the last axis (num_channels) to the
        # original "feature_size".
        outputs = {}
        outputs["SEQ_DATA"] = inputs_seq
        outputs["SEQ_LEN"] = seq_len

        return outputs
