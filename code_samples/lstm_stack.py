"""A module for defining a simple model consisting of LSTMs.

The following classes are implemented:
   * "LstmStackWithPreTraining"
"""

# pylint: disable=import-error, invalid-name, no-member, too-few-public-methods
# pylint: disable=unexpected-keyword-arg, no-name-in-module, too-many-arguments

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanw.com@samsung.com)"

# Standard imports
import copy
import platform
from enum import Enum
from dataclasses import dataclass

# Third-party imports
import numpy as np
import tensorflow as tf
from google.protobuf import any_pb2
from packaging import version

# Custom imports
from machine_learning.layers import dropout
from machine_learning.layers import layer_type
from machine_learning.layers import spec_augment_layer
from machine_learning.layers import subsampling_layer
from machine_learning.models import model_params_pb2

logger = tf.get_logger()

# At least Python 3.7 is required because of the usage of @dataclass
# At least Python 3.4 is required because of the usage of Enum.
assert version.parse(platform.python_version()) >= version.parse("3.7.0"), (
    "At least python verion 3.7 is required.")

assert version.parse(tf.__version__) >= version.parse("2.0.0"), (
    "At least tensorflow 2.0 is required.")

LayerType = layer_type.Type


class LstmType(Enum):
    """The enumeration type defining the types of masking operation."""
    ULSTM = 0
    BLSTM = 1


class TrainingState(Enum):
    PRE_TRAINING = 0
    MAIN_TRAINING = 1


@dataclass
class ModelState:
    # A dataclass storing the model state information.
    training_state: TrainingState
    num_lstm_layers: int


class LstmStackWithPreTraining(tf.keras.Model):
    """A class implementing a stack of LSTM layers with optional pre-training.

    TODO(chanw.com) Explain the pre-training scheme.

    Attributes:
        _num_examples: An int32 representing the number of processed examples.
        _training: A boolean flag representing whether the model was created
            for training.
        _pre_training: A boolean flag representing whether the pre-training
            will be used.

    Typical usage example:
        params_proto = text_format.Parse('''
            lstm_type: BLSTM
            lstm_unit_size: 512
            num_classes: 129
            pre_training_state: {
                num_examples: 300000
                initial_num_lstm_layers: 2
            }
            num_lstm_layers: 6
            num_pool_layers: 3
        ''', model_params_pb2.LstmStackWithPreTrainingParams())
    """
    def __init__(self, params_proto, **kwargs) -> None:
        super(LstmStackWithPreTraining, self).__init__()

        assert isinstance(
            params_proto, model_params_pb2.LstmStackWithPreTrainingParams
        ), ("The input parameter must be the LstmStackWithPreTrainingparams"
            " type.")
        # If "training" field exists in kwargs and if that value is false, then
        # we set self._training to be False. This is needed because the model
        # structure will be different for training and for inference. More
        # specifically, during the training phase, the model may start from the
        # pre-training state while it is not the case for the inference.
        if "training" in kwargs and not kwargs["training"]:
            self._training = False
        else:
            self._training = True

        # Checks whether pre-training will be employed or not.
        if (self._training and params_proto.HasField("pre_training_state")
                and params_proto.pre_training_state.num_examples > 0):
            self._pre_training = True
        else:
            self._pre_training = False

        self._num_examples = tf.Variable(0,
                                         dtype=tf.dtypes.int64,
                                         trainable=False)

        self._num_lstm_layers = params_proto.num_lstm_layers
        self._num_pool_layers = params_proto.num_pool_layers

        # Prepares for a pre-training case.
        if self._pre_training:
            assert (params_proto.num_lstm_layers >=
                    params_proto.pre_training_state.initial_num_lstm_layers)
            # Finds the number of examples that will be processed in each sub-state
            # during the pre-training. Note that this value will be used in
            # _find_current_state method.
            self._num_examples_for_pre_training_sub_state = float(
                params_proto.pre_training_state.num_examples /
                (params_proto.num_lstm_layers -
                 params_proto.pre_training_state.initial_num_lstm_layers + 1))
            assert self._num_examples_for_pre_training_sub_state >= 1

            self._model_training_state = ModelState(
                TrainingState.PRE_TRAINING,
                params_proto.pre_training_state.initial_num_lstm_layers)
        # Prepares for a main training case.
        else:
            self._model_training_state = ModelState(
                TrainingState.MAIN_TRAINING, self._num_lstm_layers)

        assert params_proto.pre_training_state.num_examples >= 0

        if params_proto.lstm_type == model_params_pb2.ULSTM:
            self._lstm_type = LstmType.ULSTM
        elif params_proto.lstm_type == model_params_pb2.BLSTM:
            self._lstm_type = LstmType.BLSTM
        else:
            raise ValueError("Unsupported LSTM type.")

        self._layers = []
        self._layer_types = []

        # Applies a SpecAugment layer if needed.
        if params_proto.HasField("spec_augment_params"):
            any_message = any_pb2.Any()
            any_message.Pack(params_proto)
            self._layers.append(spec_augment_layer.SpecAugment(any_message))
            self._layer_types.append(LayerType.SPEC_AUGMENT)

        if params_proto.HasField("subsampling_params"):
            factory = subsampling_layer.SubsamplingFactory()
            self._layers.append(factory.create(
                params_proto.subsampling_params))
            self._layer_types.append(LayerType.SUB_SAMPLING)

        # This for-loop iterates till len(self._lstm_layers) - 1, since the
        # final LSTM layer will be called outside this for-loop. This is
        # intentionally done to make it sure that the top layer should be an
        # LSTM layer not a Max-Pool layer.
        for l in range(max(self._num_lstm_layers, self._num_pool_layers)):
            if l < self._num_lstm_layers:
                # Adds a dropout layer.
                factory = dropout.DropoutFactory()
                self._layers.append(factory.create(
                    params_proto.dropout_params))
                self._layer_types.append(LayerType.DROPOUT)

                # Adds an LSTM layer.
                self._layers.append(
                    self._create_lstm_layer(self._lstm_type,
                                            params_proto.lstm_unit_size))
                self._layer_types.append(LayerType.LSTM)

            if l < self._num_pool_layers:
                # Adds a 2:1 MaxPool1D layer.
                self._layers.append(
                    tf.keras.layers.MaxPool1D(pool_size=2, padding="same"))
                self._layer_types.append(LayerType.MAX_POOL)

        # Adds the final dense layer.
        self._layers.append(tf.keras.layers.Dense(params_proto.num_classes))
        self._layer_types.append(LayerType.DENSE)

        logger.info("--- An LSTM Model was created. ---")
        logger.info("\nSimpleLstmStackModel: lstm_type: {0} \n"
                    "                      num_lstm_layers: {1} \n"
                    "                      lstm_unit_size: {2} \n"
                    "                      num_pool_layers: {3} \n"
                    "                      num_classes: {4}".format(
                        self._lstm_type,
                        self._model_training_state.num_lstm_layers,
                        params_proto.lstm_unit_size, self._num_pool_layers,
                        params_proto.num_classes))

        self._params_proto = params_proto

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
            A boolean flag representing whether the graph needs to be rebuilt.
        """
        if (self._model_training_state.training_state ==
                TrainingState.MAIN_TRAINING):
            return False

        assert (self._model_training_state.training_state ==
                TrainingState.PRE_TRAINING)

        state = self._find_current_state(num_examples)

        # If the new state is different from the current one, update the model.
        if state != self._model_training_state:
            self._model_training_state = state
            update_graph = True
            logger.info("--- An LSTM Model was updated. ---")
            logger.info("\nSimpleLstmStackModel: lstm_type: {0} \n"
                        "                      num_lstm_layers: {1} \n"
                        "                      lstm_unit_size: {2} \n"
                        "                      num_pool_layers: {3} \n"
                        "                      num_classes: {4}".format(
                            self._lstm_type,
                            self._model_training_state.num_lstm_layers,
                            self._params_proto.lstm_unit_size,
                            self._num_pool_layers,
                            self._params_proto.num_classes))
        else:
            update_graph = False

        self._num_examples.assign(num_examples)

        return update_graph

    def _find_current_state(self, num_examples: tf.Variable) -> ModelState:
        """Finds the current state given the number of processed examples.

        Args:
            num_examples: The number of examples processed so far.

        Returns:
            A "ModelState" data class containing the current state.
        """
        # TODO(chanw.com) Maybe need to convert the type of num_examples
        # to int in the proto buf.
        if num_examples >= tf.cast(
                self._params_proto.pre_training_state.num_examples,
                dtype=tf.dtypes.int64):
            model_state = ModelState(TrainingState.MAIN_TRAINING,
                                     self._params_proto.num_lstm_layers)

        # This means that the current state is one of the sub-state of the
        # pre-training state.
        else:
            assert self._num_examples_for_pre_training_sub_state > 0
            num_lstm_layers = (
                self._params_proto.pre_training_state.initial_num_lstm_layers +
                int(
                    float(num_examples) /
                    self._num_examples_for_pre_training_sub_state))
            num_lstm_layers = min(num_lstm_layers,
                                  self._params_proto.num_lstm_layers)

            model_state = ModelState(TrainingState.PRE_TRAINING,
                                     num_lstm_layers)

        return model_state

    def _create_lstm_layer(self, lstm_type,
                           lstm_unit_size: int) -> tf.keras.layers.Layer:
        """Creates an LSTM layer.

        Args:
            lstm_type: The type of the LSTM.
            lstm_unit_size: The unit size of this LSTM.
            dropout_params: The protocol buffer for initializing the dropout.
                Note that the dropout will be placed before the LSTM layer.

        Returns:
            A created LSTM layer.
        """
        layer = None
        if lstm_type == LstmType.BLSTM:
            layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                lstm_unit_size,
                return_sequences=True,
                recurrent_initializer="glorot_uniform"),
                                                  merge_mode="concat")
        # If the uni-directional LSTM is chosen.
        elif lstm_type == LstmType.ULSTM:
            layer = tf.keras.layers.LSTM(
                lstm_unit_size,
                return_sequences=True,
                recurrent_initializer="glorot_uniform")
        else:
            raise ValueError("Unsupported LSTM type.")

        return layer

    def call(self, inputs_dict: dict, training=None) -> dict: # yapf: disable
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
        assert (
            isinstance(inputs_dict, dict)
            and {"SEQ_DATA", "SEQ_LEN"} <= inputs_dict.keys()), (
                "The inputs_dict must be a dictionary containing \"SEQ_DATA\""
                " and \"SEQ_LEN\" as keys.")
        assert len(inputs_dict["SEQ_DATA"].shape) == 3, (
            "The rank of inputs_dict[\"SEQ_DATA\"] must be three.")

        # The statement "assert training == self._training" has an issue.
        #
        # We use the Keras model callback mechanism to write checkpoint files
        # in the CWK trainer.
        # (e.g. //speech/ck_trainer/model_checkpoint/callbacks.NBestCheckpoint).
        #
        # It internally calls self.model.save(filepath, overwrite=True,
        # options=self._options) which in turn calls,
        # outputs = model(*args, **kwargs). So, in this case, the argument of
        # the call method, "training" is False, which raises an assertion with
        # the following assert.
        # assert training == self._training

        # Applies SpecAugment if it is set in the initialization proto message
        # and the training flag is True.
        outputs = copy.copy(inputs_dict)

        lstm_index = 0
        # Applies layers except the final dense layer.
        for (layer, layer_type) in zip(self._layers, self._layer_types):
            if (layer_type == LayerType.SPEC_AUGMENT
                    or layer_type == LayerType.SUB_SAMPLING):
                outputs = layer(outputs, training=training)
            else:
                outputs_data = outputs["SEQ_DATA"]
                outputs_len = outputs["SEQ_LEN"]

                if layer_type == LayerType.DROPOUT:
                    outputs_data = layer(outputs_data, training,
                                         self._num_examples)
                elif layer_type == LayerType.LSTM:
                    # The mask parameter is specified for all the LSTM layers,
                    # since it is not propagated through the MaxPool layers.
                    if lstm_index >= self._model_training_state.num_lstm_layers:
                        continue

                    mask = tf.sequence_mask(outputs_len,
                                            maxlen=tf.shape(outputs_data)[1])
                    outputs_data = layer(outputs_data,
                                         mask=mask,
                                         training=training)
                    lstm_index += 1

                else:
                    outputs_data = layer(outputs_data, training=training)

                if layer_type == LayerType.MAX_POOL:
                    # "tf.math.ceil" is done to calculate the outputs_len, since
                    # we apply "same" as the padding option in MaxPool1D:
                    #
                    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool1D
                    outputs["SEQ_LEN"] = tf.cast(tf.math.ceil(
                        tf.math.divide(outputs_len, 2)),
                                                 dtype=tf.dtypes.int32)

                outputs["SEQ_DATA"] = outputs_data

        # Applies the final dense layer.
        outputs = {}
        outputs["SEQ_DATA"] = outputs_data
        outputs["SEQ_LEN"] = outputs_len

        return outputs
