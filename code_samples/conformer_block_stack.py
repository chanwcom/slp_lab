"""A module for defining a simple model consisting of LSTMs.

The following classes are implemented:
   * "ConformerBlockStack"
"""

# pylint: disable=import-error, invalid-name, no-member, too-few-public-methods
# pylint: disable=unexpected-keyword-arg, no-name-in-module, too-many-arguments

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanw.com@samsung.com)"

# Standard imports
import copy
import enum
import platform

# Third-party imports
import numpy as np
import tensorflow as tf
from google.protobuf import any_pb2
from google.protobuf import text_format
from packaging import version
from google.protobuf import message

# Custom imports
from machine_learning.layers import conformer_block_layer
from machine_learning.layers import dropout
from machine_learning.layers import dropout_params_pb2
from machine_learning.layers import layer_params_pb2
from machine_learning.layers import layer_type
from machine_learning.layers import spec_augment_layer
from machine_learning.layers import subsampling_layer
from machine_learning.models import conformer_block_stack_pb2
from speech.trainer.ck_trainer.util import proto_util

logger = tf.get_logger()

# At least Python 3.4 is required because of the usage of Enum.
assert version.parse(platform.python_version()) >= version.parse("3.4.0"), (
    "At least python verion 3.4 is required.")

assert version.parse(tf.__version__) >= version.parse("2.0.0"), (
    "At least tensorflow 2.0 is required.")

LayerType = layer_type.Type


# TODO(chanw.com) Needs to refactor with _create_dropout_layer in
# //machine_learning/layer/conformer_block_layer.py module.
def _create_dropout_layer(params_proto: message.Message) -> tuple: # yapf: disable
    # TODO(chanw.com) Handel all dropout params case.
    if params_proto.HasField("all_dropout_params"):
        raise ValueError("Not Implemented Yet.")
    elif params_proto.HasField("dropout_params"):
        params = params_proto.dropout_params
    # When neither of "all_dropout_params" and "dropout_params" were given.
    else:
        params = dropout_params_pb2.DropoutParams()
        params.class_name = "BaselineDropout"

    factory = dropout.DropoutFactory()
    layer = factory.create(params)
    layer_type = LayerType.DROPOUT

    return (layer, layer_type)


class ConformerBlockStack(tf.keras.Model):
    """A class implementing a Conformer Block Stack structure.

    Attributes:
        TODO(chanw.com): Adds attributes.

    Typical usage example:
        If you don't want to customize each parameter, then you may use
        "conformer_block_stack_pb2.ConformerBlockStackParams" without any
        additional parameters:

        params_proto = conformer_block_stack_pb2.ConformerBlockStackParams()
        model = conformer_block_stack.ConformerBlockStack(params_proto)


        If you want to use custom values, then you may specify the parameters
        as follows:

        params_proto = text_format.Parse('''
            spec_augment_params: {
                num_freq_masks: 2
                max_freq_mask_size: 14
                num_time_masks: 2
                max_time_mask_size: 100
            }

            subsampling_params: {
                subsampling_factor: 4
                class_name:  "Conv1DSubsampling"
                class_params: {
                    [type.googleapi.com/learning.Conv1DSubsamplingParams] {
                        num_filterbank_channels: 40
                        num_conv_channels: 256
                        kernel_size: 5
                    }
                }
            }

            conformer_block_params: {
                feed_forward_module_params: {
                    activation_type: SWISH
                    model_dim: 512
                    feedforward_dim: 2048
                    dropout_rate: 0.1
                }

                mhsa_module_params: {
                    model_dim: 512
                    num_heads: 8
                    relative_positional_embedding: False
                    dropout_rate: 0.1
                }

                convolution_module_params: {
                    conv_normalization_type: BATCH_NORM_WITH_MASK
                    activation_type: SWISH
                    model_dim: 512
                    conv_kernel_size: 31
                    dropout_rate: 0.1
                }
            }

            num_conformer_blocks: 17
        ''', conformer_block_stack_pb2.ConformerBlockStackParams())
    """
    def __init__(self, params_proto, **kwargs) -> None:
        super(ConformerBlockStack, self).__init__(kwargs)

        assert isinstance(
            params_proto,
            conformer_block_stack_pb2.ConformerBlockStackParams), (
                "The input parameter must be the ConformerBlockStackparams"
                " type.")
        self._num_examples = tf.Variable(0,
                                         dtype=tf.dtypes.int64,
                                         trainable=False)
        # Note that unlike the LstmStackWithPreTraining class in the
        # lstm_stacky.py module in the same directory, "training=True"
        # does not have any effect. In the LstmStackWithPreTraining class,
        # self._training is needed, since the model will be constructed
        # differently for training and testing because of the pre-training
        # stage.
        DEFAULT_DROPOUT_RATE = 0.1
        DEFAULT_NUM_CONFORMER_BLOCKS = 17
        DEFAULT_INPUT_DROPOUT = False

        # yapf: disable
        self._num_conformer_blocks = proto_util.get_field(
            params_proto,
            "num_conformer_blocks",
            DEFAULT_NUM_CONFORMER_BLOCKS)
        input_dropout = proto_util.get_field(
            params_proto,
            "input_dropout",
            DEFAULT_INPUT_DROPOUT)
        # yapf: enable

        self._layers = []
        self._layer_types = []

        if input_dropout:
            self._build_input_dropout_structure(params_proto)
        else:
            self._build_default_structure(params_proto)

        self._params_proto = params_proto

    def _build_default_structure(self, params_proto):
        # Applies a SpecAugment layer if needed.
        if params_proto.HasField("spec_augment_params"):
            any_message = any_pb2.Any()
            any_message.Pack(params_proto.spec_augment_params)
            self._layers.append(spec_augment_layer.SpecAugment(any_message))
            self._layer_types.append(LayerType.SPEC_AUGMENT)

        self._num_pool_layers = 0

        # Processes sub-sampling.
        if params_proto.HasField("num_pool_layers"):
            self._num_pool_layers = params_proto.num_pool_layers
            assert params_proto.num_pool_layers > 0, (
                "The number of pool layers must be positive.")
        elif params_proto.HasField("subsampling_params"):
            factory = subsampling_layer.SubsamplingFactory()
            self._layers.append(factory.create(
                params_proto.subsampling_params))
            self._layer_types.append(LayerType.SUB_SAMPLING)

        # Adds a dense layer.
        self._layers.append(
            tf.keras.layers.Dense(params_proto.conformer_block_params.
                                  mhsa_module_params.model_dim))
        self._layer_types.append(LayerType.DENSE)

        # Adds UniformDropout or DropoutLayer.
        layer, layer_type = _create_dropout_layer(params_proto)
        self._layers.append(layer)
        self._layer_types.append(layer_type)

        # Creates layers of Conformer blocks.
        self._conformer_blocks = []
        for index in range(self._num_conformer_blocks):
            self._layers.append(
                conformer_block_layer.ConformerBlock(
                    params_proto.conformer_block_params))
            self._layer_types.append(LayerType.CONFORMER_BLOCK)

            if index < self._num_pool_layers:
                self._layers.append(
                    tf.keras.layers.MaxPool1D(pool_size=2, padding="same"))
                self._layer_types.append(LayerType.MAX_POOL)

        if params_proto.HasField("num_classes"):
            assert params_proto.num_classes > 0, (
                "The number of classes should be a positive value.")
            self._layers.append(tf.keras.layers.Dense(
                params_proto.num_classes))
            self._layer_types.append(LayerType.DENSE)

        logger.info("--- A Conformer-encoder model was created. ---")
        logger.info(text_format.MessageToString(params_proto, indent=8))

    def _build_input_dropout_structure(self, params_proto):
        # Applies a SpecAugment layer if needed.
        if params_proto.HasField("spec_augment_params"):
            any_message = any_pb2.Any()
            any_message.Pack(params_proto.spec_augment_params)
            self._layers.append(spec_augment_layer.SpecAugment(any_message))
            self._layer_types.append(LayerType.SPEC_AUGMENT)

        self._num_pool_layers = 0

        # Processes sub-sampling.
        if params_proto.HasField("num_pool_layers"):
            self._num_pool_layers = params_proto.num_pool_layers
            assert params_proto.num_pool_layers > 0, (
                "The number of pool layers must be positive.")
        elif params_proto.HasField("subsampling_params"):
            factory = subsampling_layer.SubsamplingFactory()
            self._layers.append(factory.create(
                params_proto.subsampling_params))
            self._layer_types.append(LayerType.SUB_SAMPLING)

        # Adds UniformDropout or DropoutLayer.
        layer, layer_type = _create_dropout_layer(params_proto)
        self._layers.append(layer)
        self._layer_types.append(layer_type)

        # Adds a dense layer.
        self._layers.append(
            tf.keras.layers.Dense(params_proto.conformer_block_params.
                                  mhsa_module_params.model_dim))
        self._layer_types.append(LayerType.DENSE)

        # Creates layers of Conformer blocks.
        self._conformer_blocks = []
        for index in range(self._num_conformer_blocks):
            self._layers.append(
                conformer_block_layer.ConformerBlock(
                    params_proto.conformer_block_params))
            self._layer_types.append(LayerType.CONFORMER_BLOCK)

            if index < self._num_pool_layers:
                self._layers.append(
                    tf.keras.layers.MaxPool1D(pool_size=2, padding="same"))
                self._layer_types.append(LayerType.MAX_POOL)

        if params_proto.HasField("num_classes"):
            assert params_proto.num_classes > 0, (
                "The number of classes should be a positive value.")
            self._layers.append(tf.keras.layers.Dense(
                params_proto.num_classes))
            self._layer_types.append(LayerType.DENSE)

        logger.info("--- A Conformer-encoder model was created. ---")
        logger.info(text_format.MessageToString(params_proto, indent=8))

    # TODO(chanw.com) This scheme might not work when using distributed
    # training. Check whether this on-the-fly update will work with distributed
    # training.
    def model_callback(self, num_examples: tf.Variable) -> bool:
        """Updates the model structure if needed.

        This method is provided as a call-back method so that the model
        structure may be updated on-the-fly. Note that this callback must be
        called only for the training phase not for the inference phase.

        Args:
            num_examples: The number of examples processed during the training.

        Returns:
            None.
        """
        # TODO TODO(chanw.com) Apply pass-bypass using the threshold.
        self._num_examples.assign(num_examples)

        for (layer, layer_type) in zip(self._layers, self._layer_types):
            if (layer_type == LayerType.CONFORMER_BLOCK):
                layer.model_callback(num_examples)

        return False

    # yapf: disable
    def call(self, inputs_dict: dict, training: bool=True) -> dict:
        """Returns the model output given a batch of inputs.

        Args:
            inputs: A dictionary containing an acoustic feature sequence.
                The keys are as follows:
                "SEQ_DATA": The acoustic feature sequence whose shape is.
                    (batch_size, feature_len, feature_size).
                "SEQ_LEN": The length of acoust feature sequences. The shape is
                    (batch_size, 1).
            training: A flag to indicate whether this method is called for
                training.

        Returns:
            A dictionary containing the model output.
                The keys are as follows:
                "SEQ_DATA": A model output sequence whose shape is
                    (batch_size, output_len, num_classes).
                "SEQ_LEN": The length of model ouputs. The shape is
                    (batch_size, 1).
        """
        # yapf: enable
        assert isinstance(inputs_dict, dict)

        # Applies SpecAugment if it is set in the initialization proto message
        # and the training flag is True.
        outputs = copy.copy(inputs_dict)

        for (layer, layer_type) in zip(self._layers, self._layer_types):
            if (layer_type == LayerType.SPEC_AUGMENT
                    or layer_type == LayerType.SUB_SAMPLING
                    or layer_type == LayerType.CONFORMER_BLOCK):
                if layer_type == LayerType.SPEC_AUGMENT:
                    outputs = layer(outputs,
                                    training=training,
                                    num_examples=self._num_examples)
                else:
                    outputs = layer(outputs, training=training)
            else:
                outputs_data = outputs["SEQ_DATA"]
                outputs_len = outputs["SEQ_LEN"]

                if layer_type == LayerType.DROPOUT:
                    outputs_data = layer(outputs_data, training,
                                         self._num_examples)
                else:
                    outputs_data = layer(outputs_data, training=training)

                if layer_type == LayerType.MAX_POOL:
                    # "tf.math.ceil" is done to calculate the outputs_len, since
                    # we apply "same" as the padding option in MaxPool1D:
                    #
                    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool1D
                    outputs_len = tf.math.ceil(tf.math.divide(outputs_len, 2))

                outputs = {}
                outputs["SEQ_DATA"] = outputs_data
                outputs["SEQ_LEN"] = outputs_len

        return outputs
