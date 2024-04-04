from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

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

# Custom imports
from speech.trainer.ck_trainer.util import proto_util
from speech.trainer.ck_trainer.model import main_factory
from speech.trainer.ck_trainer.model import model_params_pb2
from machine_learning.models import transducer_pb2
from machine_learning.layers import layer_params_pb2

logger = tf.get_logger()


class TransducerModel(tf.keras.Model):
    def __init__(self, params_proto: any_pb2.Any, **kwargs) -> None:
        super().__init__()
        self._params_proto = params_proto

        DEFAULT_BOS_IDX = 0

        #assert self._params_proto.HasField("num_classes")
        self._bos = proto_util.get_field(self._params_proto, "bos_idx",
                                         DEFAULT_BOS_IDX)
        self._num_classes = proto_util.get_field(self._params_proto,
                                                 "num_classes")
        self._embed_dim = proto_util.get_field(self._params_proto, "embed_dim")

        encoder_params = model_params_pb2.ModelParams()
        decoder_params = model_params_pb2.ModelParams()
        self._params_proto.encoder_params.Unpack(encoder_params)
        self._params_proto.pred_network_params.Unpack(decoder_params)

        self._trans = main_factory.create_model(encoder_params)
        self._pred_input_embedding = tf.keras.layers.Dense(self._embed_dim)
        self._pred = main_factory.create_model(decoder_params)
        self._joint = JointNetwork(self._num_classes + 1)

        # TODO:
        self._use_label_for_training = True

    def _embed_pred_input(self, input_dict: dict):
        padded_input = tf.pad(input_dict["SEQ_DATA"], [[0, 0], [1, 0]],
                              constant_values=self._bos)
        padded_input = tf.one_hot(padded_input, depth=self._num_classes)

        embedding = self._pred_input_embedding(padded_input)
        embedding._kears_mask = tf.sequence_mask(input_dict["SEQ_LEN"] + 1)

        input_embedding = {
            "SEQ_DATA": self._pred_input_embedding(padded_input),
            "SEQ_LEN": input_dict["SEQ_LEN"] + 1
        }
        return input_embedding

    def call(self, inputs, training: bool = True) -> dict:  # yapf :disable
        assert isinstance(inputs[0], dict)
        assert isinstance(inputs[1], dict)

        trans_input = inputs[0]
        pred_input = self._embed_pred_input(inputs[1])

        trans_output = self._trans(trans_input, training=training)
        tf.print("pred input len: ", tf.reduce_min(pred_input["SEQ_LEN"]))
        pred_output = self._pred(pred_input, training=training)

        joint_input = {"TRANS": trans_output, "PRED": pred_output}

        return self._joint(joint_input, training=training)


class JointNetwork(tf.keras.Model):
    """Joint Network"""
    def __init__(self, vocab_size):
        super().__init__()
        self._vocab_size = vocab_size
        self._output_layer = tf.keras.layers.Dense(self._vocab_size + 1)

    def call(self, inputs_dict, training: bool=True) -> dict: # yapf: disable
        """Returns the model output given a batch of inputs.

        Args:
            inputs:
                The keys are as follows:
                "TRANS": The output dictionaty from the encoder model.
                    (batch_size, output_len, output_size).
                "PRED": The output dictionary from the prediction network.
                    (batch_size, label_len, output_size).
            training: A flag to indicate whether this method is called for
                training.

        Returns:
            A dictionary containing the model output.
                The keys are as follows:
                "SEQ_DATA": A model output sequence whose shape is
                    (batch_size, output_len, label_len, num_classes).
                "SEQ_LEN": The length of model ouputs. The shape is
                    (batch_size, 1)

        """
        tran = inputs_dict["TRANS"]["SEQ_DATA"]
        pred = inputs_dict["PRED"]["SEQ_DATA"]

        # Expands dimension for broadcasting.
        tran = tf.expand_dims(tran, axis=2)  # [B, T, 1, D]
        pred = tf.expand_dims(pred, axis=1)  # [B, 1, U, D]

        joint = tf.add(tran, pred)  # [B, T, U, D]

        out = self._output_layer(tf.math.tanh(joint))

        outputs = {}
        outputs["SEQ_DATA"] = out
        outputs["SEQ_LEN"] = inputs_dict["TRANS"]["SEQ_LEN"]

        return outputs
