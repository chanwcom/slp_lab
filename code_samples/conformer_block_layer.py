"""A module implementing a Conformer block."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanw.com@samsung.com)"

# Standard imports
import copy
import enum

# Third-party imports
import tensorflow as tf
from google.protobuf import message

# Custom imports
from speech.trainer.tf_based_end_to_end_trainer.tf_trainer.layers import attention
from speech.trainer.tf_based_end_to_end_trainer.tf_trainer.layers import normalization
from speech.trainer.ck_trainer.util import proto_util
from machine_learning.layers import conformer_block_layer_pb2
from machine_learning.layers import dropout
from machine_learning.layers import layer_type

LayerType = layer_type.Type


# TODO(chanw.com) The following needs to be re-factored. Eventually, we should
# support "dropout_params" and eventually "dropout_rate".
def _create_dropout_layer(params_proto: message.Message) -> tuple:
    """Creates a dropout layer from a params_proto.

    The "params_proto" proto-message needs to include one of the followings:
     * dropout_params
     * dropout_rate

    Args:
        params_proto: A proto-message object containing dropout info.
    Returns:
        A dropout layer
    """

    if params_proto.HasField("dropout_params"):
        factory = dropout.DropoutFactory()
        layer = factory.create(params_proto.dropout_params)
        layer_type = LayerType.DROPOUT
    else:
        layer = tf.keras.layers.Dropout(params_proto.dropout_rate)
        layer_type = LayerType.DROPOUT

    return (layer, layer_type)


class ConformerBlock(tf.keras.layers.Layer):
    """A class implementing the Conformer Block of Conformer-Transducer.

    This class is implemented based on the following paper:

    A. Gulati, J.Qin, C-C Chiu, N. Parmar, Y. Zhang, J. Yu, W. Han, S. Wang, Z.
    Zhang, Y Wu, and R. Pang, "Conformer: Convolution-augmented Transformer
    for Speech Recognition", INTERSPEECH 2020.

    We also referred to the following papers as well.

    Q. Zhang, H. Lu, H. Sak, A. Tripathi, E. McDermott, S. Koo, and S. Kumar,
    "Transformer transducer: a streamable speech recognition model with
    transformer encoders and rnn-t loss", ICASSP 2020.
    """
    def __init__(
            self, params_proto: conformer_block_layer_pb2.ConformerBlockParams
    ) -> None:
        super(ConformerBlock, self).__init__()

        self._feed_forward_module_0 = FeedForwardModule(
            params_proto.feed_forward_module_params)

        self._mhsa_module = MHSAModule(params_proto.mhsa_module_params)

        self._convolution_module = ConvolutionModule(
            params_proto.convolution_module_params)

        self._feed_forward_module_1 = FeedForwardModule(
            params_proto.feed_forward_module_params)

        self._layer_norm_layer, __ = create_normalization_layer(
            conformer_block_layer_pb2.LAYER_NORM)

    def model_callback(self, num_examples: tf.Variable) -> None:
        """Updates the model structure if needed.

        This method is provided as a call-back method so that the model
        structure may be updated on-the-fly. Note that this callback must be
        called only during the training phase not during the testing phase.

        Args:
            num_examples: The number of examples processed during the training.

        Returns:
            None.
        """
        self._feed_forward_module_0.model_callback(num_examples)
        self._mhsa_module.model_callback(num_examples)
        self._convolution_module.model_callback(num_examples)
        self._feed_forward_module_1.model_callback(num_examples)

    # yapf: disable
    def call(self, inputs_dict: dict, training: bool=None) -> dict:
        """Applies the ConformerBlock to a batch of input sequences."""
        # yapf: enable

        assert (
            isinstance(inputs_dict, dict)
            and {"SEQ_DATA", "SEQ_LEN"} <= inputs_dict.keys()), (
                "The inputs_dict must be a dictionary containing \"SEQ_DATA\""
                " and \"SEQ_LEN\" as keys.")
        assert len(inputs_dict["SEQ_DATA"].shape) == 3, (
            "The rank of inputs_dict[\"SEQ_DATA\"] must be three.")

        outputs_dict = copy.copy(inputs_dict)

        # yapf: disable
        outputs_dict["SEQ_DATA"] = (
            0.5 * self._feed_forward_module_0(
                outputs_dict, training=training)["SEQ_DATA"]
            + outputs_dict["SEQ_DATA"])

        outputs_dict["SEQ_DATA"] = (
            self._mhsa_module(outputs_dict, training=training)["SEQ_DATA"]
            + outputs_dict["SEQ_DATA"])

        outputs_dict["SEQ_DATA"] = (
            self._convolution_module(
                outputs_dict, training=training)["SEQ_DATA"]
            + outputs_dict["SEQ_DATA"])

        outputs_dict["SEQ_DATA"] = (
            0.5 * self._feed_forward_module_1(
                outputs_dict, training=training)["SEQ_DATA"]
            + outputs_dict["SEQ_DATA"])

        outputs_dict["SEQ_DATA"] = (
            self._layer_norm_layer(
                outputs_dict["SEQ_DATA"], training=training))
        # yapf: enable

        return outputs_dict


class GLU(tf.keras.layers.Layer):
    """A Keras class implementing the Gated Linear Unit (GLU).

    As of Tensorflow version 2.9, Keras does not have a built-in API for GLU.
    This class is implemented to be employed for the Convolution block in the
    Conformer-Transducer paper: TODO(chanw.com) Add the paper link.


    The following code was refereed:
    * https://pytorch.org/docs/stable/generated/torch.nn.functional.glu.html
    * https://github.com/TensorSpeech/TensorFlowASR/blob/main/tensorflow_asr/models/activations/glu.py
    """
    def __init__(
            self,
            axis=-1,
            name: str="glu_activation",
            **kwargs) -> None:  # yapf: disable
        super(GLU, self).__init__(name=name, **kwargs)
        self._axis = axis

    def call(self, inputs, **kwargs):
        a, b = tf.split(inputs, 2, axis=self._axis)
        return tf.multiply(a, tf.nn.sigmoid(b))


def create_normalization_layer(layer_type: int) -> tuple:
    """Creates various normalization Keras layers.

    Args:
        layer_type: An integer type defined by "NormalizationType".
            "NormalizationType" is an enumeration type defined in
            "layer_params.proto".

    Returns:
        A tuple containing the layer object and the corresponding type.
    """
    # TODO(chanw.com) Support more normalizations type such as
    # GroupNormalization and so on.
    if layer_type == conformer_block_layer_pb2.LAYER_NORM:
        layer = tf.keras.layers.LayerNormalization()
        layer_type = LayerType.LAYER_NORM
    elif layer_type == conformer_block_layer_pb2.BATCH_NORM_WITH_MASK:
        layer = normalization.BatchNormWithMask(sync=True)
        layer_type = LayerType.BATCH_NORM_WITH_MASK
    else:
        raise ValueError("Unsupported normalization type.")

    return (layer, layer_type)


def _activation_str(proto_type) -> str:
    """Returns the string given the activation enumeration type."""
    if proto_type == conformer_block_layer_pb2.SWISH:
        activation = "swish"
    elif proto_type == conformer_block_layer_pb2.RELU:
        activation = "relu"
    else:
        raise ValueError("Unsupported activation type.")

    return activation


class FeedForwardModule(tf.keras.layers.Layer):
    """A class implementing the feed-forward block of a Transformer.

    Specifically, this implementation has been motivated by FeedForward block
    of the conformer-Transducer.

    A. Gulati, J.Qin, C-C Chiu, N. Parmar, Y. Zhang, J. Yu, W. Han, S. Wang, Z.
    Zhang, Y Wu, and R. Pang, "Conformer: Convolution-augmented Transformer
    for Speech Recognition", INTERSPEECH 2020.

    We also refer to the following paper, but the structure is not exactly the
    same.
    TODO(chanw.com) Modify the FeedForwardModule to support the different types.

    Q. Zhang, H. Lu, H. Sak, A. Tripathi, E. McDermott, S. Koo, and S. Kumar,
    "Transformer transducer: a streamable speech recognition model with
    transformer encoders and rnn-t loss", ICASSP 2020.

    """
    def __init__(
        self, params_proto: conformer_block_layer_pb2.FeedForwardModuleParams
    ) -> None:
        super(FeedForwardModule, self).__init__()

        self._layers = []
        self._layer_types = []

        self._num_examples = tf.Variable(0,
                                         dtype=tf.dtypes.int64,
                                         trainable=False)

        # Defines the default values.
        DEFAULT_ACTIVATIN_TYPE = conformer_block_layer_pb2.SWISH
        DEFAULT_MODEL_DIM = 512
        DEFAULT_FEEDFORWARD_DIM = 2048
        DEFAULT_DROPOUT_RATE = 0.1
        DEFAULT_INPUT_DROPOUT = False

        # yapf: disable
        self._activation_type = proto_util.get_field(
            params_proto,
            "activation_type",
            DEFAULT_ACTIVATIN_TYPE)
        self._model_dim = proto_util.get_field(
            params_proto,
            "model_dim",
            DEFAULT_MODEL_DIM)
        self._feedforward_dim = proto_util.get_field(
            params_proto,
            "feedforward_dim",
            DEFAULT_FEEDFORWARD_DIM)
        self._dropout_rate = proto_util.get_field(
            params_proto,
            "dropout_rate",
            DEFAULT_DROPOUT_RATE)
        input_dropout = proto_util.get_field(
            params_proto,
            "input_dropout",
            DEFAULT_INPUT_DROPOUT)
        # yapf: enable

        if input_dropout:
            self._build_input_dropout_structure(params_proto)
        else:
            self._build_default_structure(params_proto)

        self._params_proto = params_proto

    def _build_default_structure(self, params_proto):
        layer, layer_type = create_normalization_layer(
            conformer_block_layer_pb2.LAYER_NORM)
        self._layers.append(layer)
        self._layer_types.append(layer_type)

        self._layers.append(tf.keras.layers.Dense(units=self._feedforward_dim))
        self._layer_types.append(LayerType.DENSE)

        self._layers.append(
            tf.keras.layers.Activation(_activation_str(self._activation_type)))
        self._layer_types.append(LayerType.ACTIVATION)

        # Adds a DropoutLayer.
        layer, layer_type = _create_dropout_layer(params_proto)
        self._layers.append(layer)
        self._layer_types.append(layer_type)

        self._layers.append(tf.keras.layers.Dense(units=self._model_dim))
        self._layer_types.append(LayerType.DENSE)

        # Adds a dropout layer.
        layer, layer_type = _create_dropout_layer(params_proto)
        self._layers.append(layer)
        self._layer_types.append(layer_type)

    def _build_input_dropout_structure(self, params_proto):
        layer, layer_type = create_normalization_layer(
            conformer_block_layer_pb2.LAYER_NORM)
        self._layers.append(layer)
        self._layer_types.append(layer_type)

        # Adds a dropout layer.
        layer, layer_type = _create_dropout_layer(params_proto)
        self._layers.append(layer)
        self._layer_types.append(layer_type)

        self._layers.append(tf.keras.layers.Dense(units=self._feedforward_dim))
        self._layer_types.append(LayerType.DENSE)

        self._layers.append(
            tf.keras.layers.Activation(_activation_str(self._activation_type)))
        self._layer_types.append(LayerType.ACTIVATION)

        # Adds a dropout layer.
        layer, layer_type = _create_dropout_layer(params_proto)
        self._layers.append(layer)
        self._layer_types.append(layer_type)

        self._layers.append(tf.keras.layers.Dense(units=self._model_dim))
        self._layer_types.append(LayerType.DENSE)

    def model_callback(self, num_examples: tf.Variable) -> None:
        """Updates the model structure if needed.

        This method is provided as a call-back method so that the model
        structure may be updated on-the-fly. Note that this callback must be
        called only during the training phase not during the testing phase.

        Args:
            num_examples: The number of examples processed during the training.

        Returns:
            None.
        """
        self._num_examples.assign(num_examples)

    # yapf: disable
    def call(self, inputs_dict: dict, training: bool=None) -> dict:
        """Applies the FeedForwardModule to a batch of input sequences."""
        # yapf: enable

        assert (
            isinstance(inputs_dict, dict)
            and {"SEQ_DATA", "SEQ_LEN"} <= inputs_dict.keys()), (
                "The inputs_dict must be a dictionary containing \"SEQ_DATA\""
                " and \"SEQ_LEN\" as keys.")
        assert len(inputs_dict["SEQ_DATA"].shape) == 3, (
            "The rank of inputs_dict[\"SEQ_DATA\"] must be three.")

        inputs_data = copy.copy(inputs_dict["SEQ_DATA"])

        mask = tf.cast(tf.sequence_mask(inputs_dict["SEQ_LEN"],
                                        maxlen=tf.shape(inputs_data)[1]),
                       dtype=inputs_data.dtype)

        assert len(self._layers) == len(self._layer_types)

        for (layer, layer_type) in zip(self._layers, self._layer_types):
            if layer_type == LayerType.BATCH_NORM_WITH_MASK:
                inputs_data = layer(inputs_data, mask=mask, training=training)
            elif layer_type == LayerType.DROPOUT:
                inputs_data = layer(inputs_data, training, self._num_examples)
            else:
                inputs_data = layer(inputs_data, training=training)

        outputs = {}

        outputs["SEQ_DATA"] = tf.expand_dims(mask, axis=2) * inputs_data
        outputs["SEQ_LEN"] = inputs_dict["SEQ_LEN"]

        return outputs


class MHSAModule(tf.keras.layers.Layer):
    """A class implementing the Multi-Head Self Attention Module (MHSA).

    The implementation is based on the following paper.
    A. Gulati, J.Qin, C-C Chiu, N. Parmar, Y. Zhang, J. Yu, W. Han, S. Wang, Z.
    Zhang, Y Wu, and R. Pang, "Conformer: Convolution-augmented Transformer
    for Speech Recognition", INTERSPEECH-2020. 2020, pp.5036-5040.
    """
    def __init__(
            self,
            params_proto: conformer_block_layer_pb2.MHSAModuleParams) -> None:
        super(MHSAModule, self).__init__()

        self._layers = []
        self._layer_types = []

        self._num_examples = tf.Variable(0,
                                         dtype=tf.dtypes.int64,
                                         trainable=False)

        # Defines the default values.
        DEFAULT_MODEL_DIM = 512
        DEFAULT_NUM_HEADS = 8
        DEFAULT_RELATIVE_POSITIONAL_EMBEDDING = True
        DEFAULT_DROPOUT_RATE = 0.1
        DEFAULT_MASK = -1
        DEFAULT_CAUSALITY = False
        DEFAULT_INPUT_DROPOUT = False

        # yapf: disable
        self._model_dim = proto_util.get_field(
            params_proto,
            "model_dim",
            DEFAULT_MODEL_DIM)
        self._num_heads = proto_util.get_field(
            params_proto,
            "num_heads",
            DEFAULT_NUM_HEADS)
        self._relative_positional_embedding = proto_util.get_field(
            params_proto,
            "relative_positional_embedding",
            DEFAULT_RELATIVE_POSITIONAL_EMBEDDING)
        self._dropout_rate = proto_util.get_field(
            params_proto,
            "dropout_rate",
            DEFAULT_DROPOUT_RATE)
        self._left_mask = proto_util.get_field(
            params_proto,
            "left_mask",
            DEFAULT_MASK)
        self._right_mask = proto_util.get_field(
            params_proto,
            "right_mask",
            DEFAULT_MASK)
        self._causal = proto_util.get_field(
            params_proto,
            "causal",
            DEFAULT_CAUSALITY)
        input_dropout = proto_util.get_field(
            params_proto,
            "input_dropout",
            DEFAULT_INPUT_DROPOUT)
        # yapf: enable

        if input_dropout:
            self._build_input_dropout_structure(params_proto)
        else:
            self._build_default_structure(params_proto)

        self._params_proto = params_proto

    def _build_default_structure(self, params_proto):
        layer, layer_type = create_normalization_layer(
            conformer_block_layer_pb2.LAYER_NORM)
        self._layers.append(layer)
        self._layer_types.append(layer_type)

        # We use the Multi-Head Attention with the relative positional
        # embedding by default. Refer to Section 2.1 of the following
        # paper for more detail:
        #
        # A. Gulati, J.Qin, C-C Chiu, N. Parmar, Y. Zhang, J. Yu,
        # W. Han, S. Wang, Z. Zhang, Y Wu, and R. Pang,
        # "Conformer: Convolution-augmented Transformer for Speech
        # Recognition", INTERSPEECH 2020.
        if self._relative_positional_embedding:
            self._layers.append(
                attention.RelativeAttention(att_dim=self._model_dim,
                                            att_head=self._num_heads,
                                            input_dim=self._model_dim,
                                            left_mask=self._left_mask,
                                            right_mask=self._right_mask))
        else:
            self._layers.append(
                tf.keras.layers.MultiHeadAttention(num_heads=self._num_heads,
                                                   key_dim=self._model_dim))

        self._layer_types.append(LayerType.MULTI_HEAD_ATTENTION)

        # Adds a dropout layer.
        layer, layer_type = _create_dropout_layer(params_proto)
        self._layers.append(layer)
        self._layer_types.append(layer_type)

    def _build_input_dropout_structure(self, params_proto):
        layer, layer_type = create_normalization_layer(
            conformer_block_layer_pb2.LAYER_NORM)
        self._layers.append(layer)
        self._layer_types.append(layer_type)

        # Adds a dropout layer.
        layer, layer_type = _create_dropout_layer(params_proto)
        self._layers.append(layer)
        self._layer_types.append(layer_type)

        # We use the Multi-Head Attention with the relative positional
        # embedding by default. Refer to Section 2.1 of the following
        # paper for more detail:
        #
        # A. Gulati, J.Qin, C-C Chiu, N. Parmar, Y. Zhang, J. Yu,
        # W. Han, S. Wang, Z. Zhang, Y Wu, and R. Pang,
        # "Conformer: Convolution-augmented Transformer for Speech
        # Recognition", INTERSPEECH 2020.
        if self._relative_positional_embedding:
            self._layers.append(
                attention.RelativeAttention(att_dim=self._model_dim,
                                            att_head=self._num_heads,
                                            input_dim=self._model_dim,
                                            left_mask=self._left_mask,
                                            right_mask=self._right_mask))
        else:
            self._layers.append(
                tf.keras.layers.MultiHeadAttention(num_heads=self._num_heads,
                                                   key_dim=self._model_dim))

        self._layer_types.append(LayerType.MULTI_HEAD_ATTENTION)

    def model_callback(self, num_examples: tf.Variable) -> None:
        """Updates the model structure if needed.

        This method is provided as a call-back method so that the model
        structure may be updated on-the-fly. Note that this callback must be
        called only during the training phase not during the testing phase.

        Args:
            num_examples: The number of examples processed during the training.

        Returns:
            None.
        """
        self._num_examples.assign(num_examples)

    # yapf: disable
    def call(self, inputs_dict: dict, training: bool=None) -> dict:
        """Applies the FeedForwardModule to a batch of input sequences."""
        # yapf: enable

        assert (
            isinstance(inputs_dict, dict)
            and {"SEQ_DATA", "SEQ_LEN"} <= inputs_dict.keys()), (
                "The inputs_dict must be a dictionary containing \"SEQ_DATA\""
                " and \"SEQ_LEN\" as keys.")
        assert len(inputs_dict["SEQ_DATA"].shape) == 3, (
            "The rank of inputs_dict[\"SEQ_DATA\"] must be three.")

        inputs_data = copy.copy(inputs_dict["SEQ_DATA"])

        mask = tf.cast(tf.sequence_mask(inputs_dict["SEQ_LEN"],
                                        maxlen=tf.shape(inputs_data)[1]),
                       dtype=inputs_data.dtype)

        assert len(self._layers) == len(self._layer_types)

        for (layer, layer_type) in zip(self._layers, self._layer_types):
            if layer_type == LayerType.MULTI_HEAD_ATTENTION:
                # We use the Multi-Head Attention with the relative positional
                # embedding by default. Refer to Section 2.1 of the following
                # paper for more detail:
                #
                # A. Gulati, J.Qin, C-C Chiu, N. Parmar, Y. Zhang, J. Yu,
                # W. Han, S. Wang, Z. Zhang, Y Wu, and R. Pang,
                # "Conformer: Convolution-augmented Transformer for Speech
                # Recognition", INTERSPEECH 2020.
                if self._relative_positional_embedding:
                    inputs_data = layer(inputs_data,
                                        inputs_dict["SEQ_LEN"],
                                        training=training)
                else:
                    attention_mask = mask[:, tf.newaxis, tf.newaxis, :]
                    inputs_data = layer(query=inputs_data,
                                        value=inputs_data,
                                        attention_mask=attention_mask,
                                        use_causal_mask=self._causal,
                                        training=training)
            elif layer_type == LayerType.DROPOUT:
                inputs_data = layer(inputs_data, training, self._num_examples)
            else:
                inputs_data = layer(inputs_data, training=training)

        outputs = {}

        outputs["SEQ_DATA"] = tf.expand_dims(mask, axis=2) * inputs_data
        outputs["SEQ_LEN"] = inputs_dict["SEQ_LEN"]

        return outputs


class ConvolutionModule(tf.keras.layers.Layer):
    """A class implementing the feed-forward block of a Conformer-Transducer.

    A. Gulati, J.Qin, C-C Chiu, N. Parmar, Y. Zhang, J. Yu, W. Han, S. Wang, Z.
    Zhang, Y Wu, and R. Pang, "Conformer: Convolution-augmented Transformer
    for Speech Recognition", INTERSPEECH 2020.
    """
    def __init__(
        self, params_proto: conformer_block_layer_pb2.ConvolutionModuleParams
    ) -> None:
        super(ConvolutionModule, self).__init__()

        self._layers = []
        self._layer_types = []

        self._num_examples = tf.Variable(0,
                                         dtype=tf.dtypes.int64,
                                         trainable=False)

        # Defines the default values.
        DEFAULT_NORMALIZATION_TYPE = conformer_block_layer_pb2.BATCH_NORM_WITH_MASK
        DEFAULT_ACTIVATIN_TYPE = conformer_block_layer_pb2.SWISH
        DEFAULT_MODEL_DIM = 512
        DEFAULT_CONV_KERNEL_SIZE = 31
        DEFAULT_DROPOUT_RATE = 0.1
        DEFAULT_CAUSAL = False
        DEFAULT_INPUT_DROPOUT = False

        # yapf: disable
        self._normalization_type = proto_util.get_field(
            params_proto,
            "conv_normalization_type",
            DEFAULT_NORMALIZATION_TYPE)
        self._activation_type = proto_util.get_field(
            params_proto,
            "activation_type",
            DEFAULT_ACTIVATIN_TYPE)
        self._model_dim = proto_util.get_field(
            params_proto,
            "model_dim",
            DEFAULT_MODEL_DIM)
        self._conv_kernel_size = proto_util.get_field(
            params_proto,
            "conv_kernel_size",
            DEFAULT_CONV_KERNEL_SIZE)
        self._dropout_rate = proto_util.get_field(
            params_proto,
            "dropout_rate",
            DEFAULT_DROPOUT_RATE)
        self._causal = proto_util.get_field(
            params_proto,
            "causal",
            DEFAULT_CAUSAL)
        input_dropout = proto_util.get_field(
            params_proto,
            "input_dropout",
            DEFAULT_INPUT_DROPOUT)
        # yapf: enable

        if input_dropout:
            self._build_input_dropout_structure(params_proto)
        else:
            self._build_default_structure(params_proto)

        self._params_proto = params_proto

    def _build_default_structure(self, params_proto):
        layer, layer_type = create_normalization_layer(
            conformer_block_layer_pb2.LAYER_NORM)
        self._layers.append(layer)
        self._layer_types.append(layer_type)

        self._layers.append(
            tf.keras.layers.Dense(units=int(self._model_dim * 2)))
        self._layer_types.append(LayerType.DENSE)

        self._layers.append(GLU())
        self._layer_types.append(LayerType.ACTIVATION)

        # Adds the masking layer so that tf.keras.layers.DepthwiseConv1D is not
        # affected by the padded portion.
        #
        # Refer to the following link for more detailed information.
        # https://github.com/keras-team/keras/issues/411
        self._layers.append(tf.keras.layers.Masking(mask_value=0.0))
        self._layer_types.append(LayerType.MASKING)

        if not self._causal:
            self._layers.append(
                tf.keras.layers.DepthwiseConv1D(
                    kernel_size=self._conv_kernel_size, padding="same"))
            self._layer_types.append(LayerType.DEPTHWISE_CONV1D)
        else:
            self._layers.append(
                tf.keras.layers.ZeroPadding1D(padding=(self._conv_kernel_size -
                                                       1, 0)))
            self._layers.append(
                tf.keras.layers.DepthwiseConv1D(
                    kernel_size=self._conv_kernel_size, padding="valid"))

            self._layer_types.append(LayerType.PADDING)
            self._layer_types.append(LayerType.DEPTHWISE_CONV1D)

        layer, layer_type = create_normalization_layer(
            self._normalization_type)
        self._layers.append(layer)
        self._layer_types.append(layer_type)

        self._layers.append(
            tf.keras.layers.Activation(_activation_str(self._activation_type)))
        self._layer_types.append(LayerType.ACTIVATION)

        self._layers.append(tf.keras.layers.Dense(units=self._model_dim))
        self._layer_types.append(LayerType.DENSE)

        # Adds a dropout layer.
        layer, layer_type = _create_dropout_layer(params_proto)
        self._layers.append(layer)
        self._layer_types.append(layer_type)

    def _build_input_dropout_structure(self, params_proto):
        layer, layer_type = create_normalization_layer(
            conformer_block_layer_pb2.LAYER_NORM)
        self._layers.append(layer)
        self._layer_types.append(layer_type)

        # Adds a dropout layer.
        layer, layer_type = _create_dropout_layer(params_proto)
        self._layers.append(layer)
        self._layer_types.append(layer_type)

        self._layers.append(
            tf.keras.layers.Dense(units=int(self._model_dim * 2)))
        self._layer_types.append(LayerType.DENSE)

        self._layers.append(GLU())
        self._layer_types.append(LayerType.ACTIVATION)

        # Adds the masking layer so that tf.keras.layers.DepthwiseConv1D is not
        # affected by the padded portion.
        #
        # Refer to the following link for more detailed information.
        # https://github.com/keras-team/keras/issues/411
        self._layers.append(tf.keras.layers.Masking(mask_value=0.0))
        self._layer_types.append(LayerType.MASKING)

        # Adds a dropout layer.
        layer, layer_type = _create_dropout_layer(params_proto)
        self._layers.append(layer)
        self._layer_types.append(layer_type)

        if not self._causal:
            self._layers.append(
                tf.keras.layers.DepthwiseConv1D(
                    kernel_size=self._conv_kernel_size, padding="same"))
            self._layer_types.append(LayerType.DEPTHWISE_CONV1D)
        else:
            self._layers.append(
                tf.keras.layers.ZeroPadding1D(padding=(self._conv_kernel_size -
                                                       1, 0)))
            self._layers.append(
                tf.keras.layers.DepthwiseConv1D(
                    kernel_size=self._conv_kernel_size, padding="valid"))

            self._layer_types.append(LayerType.PADDING)
            self._layer_types.append(LayerType.DEPTHWISE_CONV1D)

        layer, layer_type = create_normalization_layer(
            self._normalization_type)
        self._layers.append(layer)
        self._layer_types.append(layer_type)

        self._layers.append(
            tf.keras.layers.Activation(_activation_str(self._activation_type)))
        self._layer_types.append(LayerType.ACTIVATION)

        # Adds a dropout layer.
        layer, layer_type = _create_dropout_layer(params_proto)
        self._layers.append(layer)
        self._layer_types.append(layer_type)

        self._layers.append(tf.keras.layers.Dense(units=self._model_dim))
        self._layer_types.append(LayerType.DENSE)

    def model_callback(self, num_examples: tf.Variable) -> None:
        """Updates the model structure if needed.

        This method is provided as a call-back method so that the model
        structure may be updated on-the-fly. Note that this callback must be
        called only during the training phase not during the testing phase.

        Args:
            num_examples: The number of examples processed during the training.

        Returns:
            None.
        """
        self._num_examples.assign(num_examples)

    # yapf: disable
    def call(self, inputs_dict: dict, training: bool=None) -> dict:
        """Applies the FeedForwardModule to a batch of input sequences."""
        # yapf: enable

        assert (
            isinstance(inputs_dict, dict)
            and {"SEQ_DATA", "SEQ_LEN"} <= inputs_dict.keys()), (
                "The inputs_dict must be a dictionary containing \"SEQ_DATA\""
                " and \"SEQ_LEN\" as keys.")
        assert len(inputs_dict["SEQ_DATA"].shape) == 3, (
            "The rank of inputs_dict[\"SEQ_DATA\"] must be three.")

        inputs_data = copy.copy(inputs_dict["SEQ_DATA"])

        mask = tf.cast(tf.sequence_mask(inputs_dict["SEQ_LEN"],
                                        maxlen=tf.shape(inputs_data)[1]),
                       dtype=inputs_data.dtype)

        assert len(self._layers) == len(self._layer_types)

        for (layer, layer_type) in zip(self._layers, self._layer_types):
            if layer_type == LayerType.BATCH_NORM_WITH_MASK:
                inputs_data = layer(inputs_data, mask=mask, training=training)
            elif layer_type == LayerType.DROPOUT:
                inputs_data = layer(inputs_data, training, self._num_examples)
            else:
                inputs_data = layer(inputs_data, training=training)

        outputs = {}
        outputs["SEQ_DATA"] = tf.expand_dims(mask, axis=2) * inputs_data
        outputs["SEQ_LEN"] = inputs_dict["SEQ_LEN"]

        return outputs
