"""A module implementing classes derived from Tf2DatasetOperation.

Implemented operations:
 * BasicDatasetOperation
 * DatasetFilterOperation
 * DatasetWrapperOperation
 * DatasetTypeConversionOperation
 * DictToTupleDatasetOperation
 * UtteranceDataPreprocessor
"""

# pylint: disable=invalid-name, no-member, import-error
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanw.com@samsung.com)"

# Standard imports
import abc

# Third party imports
import tensorflow as tf
from google.protobuf import any_pb2
from google.protobuf import message

# Custom imports
from math_lib.operation import main_factory
from math_lib.operation import operation
from speech.trainer.ck_trainer.dataset_operation import dataset_operation_params_pb2
from util import proto_util


class Tf2DatasetOperation(operation.AbstractOperation):
    """An abstract class for TF2_DATASET type."""
    @abc.abstractmethod
    def __init__(self, params_proto, params_dict=None, operation_dict=None):
        super(Tf2DatasetOperation, self).__init__(params_proto)
        pass

    @abc.abstractmethod
    def process(self, dataset):
        """Processes an input dataset and returns the processed output.

        Args:
            dataset: A tf.data.Dataset input to this operation.

        Returns:
            A processed tf.data.Dataset object.

        """
        if not isinstance(dataset, tf.data.Dataset):
            raise ValueError("The type of dataset must be tf.data.Dataset.")


class BasicDatasetOperation(Tf2DatasetOperation):
    """A class implementing a basic dataset operation."""
    def __init__(self,
                 params_proto: message.Message,
                 params_dict: dict=None,
                 operation_dict: dict=None) -> None:  # yapf: disable
        super(BasicDatasetOperation, self).__init__(params_proto)
        # yapf: disable
        # Checks the input argument type.
        assert (isinstance(params_proto, any_pb2.Any) or
                isinstance(params_proto,
                    dataset_operation_params_pb2.BasicDatasetOperationParams))
        # yapf: enable

        # Unpacks "params_proto" if the type is any_pb2.Any.
        self._params_proto = proto_util.maybe_unpack(
            params_proto,
            dataset_operation_params_pb2.BasicDatasetOperationParams)

    def process(self, dataset):
        """Processes the input dataset and returns the processed output."""
        if not isinstance(dataset, tf.data.Dataset):
            raise ValueError("The type of dataset must be tf.data.Dataset.")

        if self._params_proto.HasField("batch_size"):
            # TODO(chanw.com) Separate padded_batch with batch.
            dataset = dataset.padded_batch(self._params_proto.batch_size)
            assert not self._params_proto.unbatch, (
                "batch_size and unbatch cnanot be specified simultaneously.")

        if self._params_proto.unbatch:
            dataset = dataset.unbatch()

        if self._params_proto.use_cache:
            dataset = dataset.cache()

        if self._params_proto.use_prefetch:
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset


#class SeqDataUnbatch(Tf2DatasetOperation):
#    def
#
#    def process(self, dataset):
#        dataset = dataset.unbatch()
#
#        dataset = dataset.map(


class DatasetFilterOperation(Tf2DatasetOperation):
    """A class implementing dataset filter operation."""
    def __init__(self,
                 params_proto: any_pb2.Any,
                 params_dict: dict=None,
                 operation_dict: dict=None) -> None:  # yapf: disable
        super(DatasetFilterOperation, self).__init__(params_proto)

        # yapf: disable
        # Checks the input argument type.
        assert (isinstance(params_proto, any_pb2.Any) or
                isinstance(params_proto,
                    dataset_operation_params_pb2.DatasetFilterOperationParams))
        # yapf: enable

        self._params_proto = proto_util.maybe_unpack(
            params_proto,
            dataset_operation_params_pb2.DatasetFilterOperationParams)

        factory = main_factory.Factory()
        self._operation = factory.create_operation(
            self._params_proto.operation_params)

    def process(self, dataset):
        assert isinstance(dataset, tf.data.Dataset)

        return dataset.filter(self._operation.process)


class SequenceBatchOperation(Tf2DatasetOperation):
    # TODO(chanw.com) Implement this class with a suitable unit test.
    pass


class DatasetWrapperOperation(Tf2DatasetOperation):
    """A class for wrapping operations to make a DatasetOperation.

    Note that with conventional "Operations", the inputs and outputs
    are usually Tensors or NumPy arrays. With the DatasetOperation, the
    inputs and outputs are TF2 Dataset.

    Example:

        In the following case, a standard operation OperationDouble
        is wrapped as a DatasetOperation.

        params_proto = text_format.Parse(
            '''
            [type.googleapi.com/learning.DatasetWrapperOperationParams] {
                operation_params: {
                    class_name: "OperationDouble"
                    class_params: {
                        [type.googleapi.com/learning.OperationDoubleParams] {
                        }
                    }
                }
            }
        ''', any_pb2.Any())

        operation = tf2_dataset_operation.DatasetWrapperOperation(params_proto)
        output_dataset = operation.process(self._input_dataset)


    """
    def __init__(self, params_proto, params_dict=None, operation_dict=None):
        super(DatasetWrapperOperation,
              self).__init__(params_proto, params_dict, operation_dict)

        # Checks the input argument type.
        assert isinstance(params_proto, any_pb2.Any)

        class_params = (
            dataset_operation_params_pb2.DatasetWrapperOperationParams())
        params_proto.Unpack(class_params)

        factory = main_factory.Factory()
        self._operation = factory.create_operation(
            class_params.operation_params)

    def process(self, dataset):
        assert isinstance(dataset, tf.data.Dataset)

        return dataset.map(lambda examples: self._operation.process(examples),
                           num_parallel_calls=tf.data.experimental.AUTOTUNE)


class DatasetTypeConversionOperation(Tf2DatasetOperation):
    """An operation class for converting the type."""
    def __init__(self,
                 any_params_proto,
                 params_dict=None,
                 operation_dict=None):
        super(DatasetTypeConversionOperation,
              self).__init__(any_params_proto, params_dict, operation_dict)

        # Unpacks the any proto message.
        assert isinstance(any_params_proto, any_pb2.Any)
        self._params_proto = (dataset_operation_params_pb2.
                              DatasetTypeConversionOperationParams())
        any_params_proto.Unpack(self._params_proto)

        self._in_type = self._enum_to_tf_dtype(self._params_proto.in_type)
        self._out_type = self._enum_to_tf_dtype(self._params_proto.out_type)

        # The type must be either a floating-point type or an integer type.
        #
        # Examples that do not satisfy this requirement include a string type,
        # a boolean type, and so on.
        assert self._in_type.is_floating or self._in_type.is_integer

    def _enum_to_tf_dtype(self, enum_type):
        dtype = (
            dataset_operation_params_pb2.DatasetTypeConversionOperationParams)

        if enum_type == dtype.UINT8:
            return tf.dtypes.uint8
        elif enum_type == dtype.UINT16:
            return tf.dtypes.uint16
        elif enum_type == dtype.UINT32:
            return tf.dtypes.uint32
        elif enum_type == dtype.INT8:
            return tf.dtypes.int8
        elif enum_type == dtype.INT16:
            return tf.dtypes.int16
        elif enum_type == dtype.INT32:
            return tf.dtypes.int32
        elif enum_type == dtype.FLOAT16:
            return tf.dtypes.float16
        elif enum_type == dtype.FLOAT32:
            return tf.dtypes.float32
        elif enum_type == dtype.FLOAT64:
            return tf.dtypes.float64
        else:
            raise TypeError("Unsupported types.")

    def _convert_type(self, data):
        # In this case, the data is a two-tuple structure (inputs, labels).
        if self._params_proto.as_supervised:
            try:
                assert len(data) == 2
            except:
                raise Exception("The data is given in a supervised format.")

            inputs = data[0]
            labels = data[1]
        else:
            inputs = data

        # Checks whether the input type is correct.
        assert self._in_type == inputs.dtype

        # If the input and output types are the same, just returns the input.
        if self._in_type == self._out_type:
            return data

        # Converts from a floating-point type to another floating-point type.
        if self._in_type.is_floating and self._out_type.is_floating:
            inputs = tf.cast(inputs, dtype=self._out_type)

        # Converts from a floating-point type to an integer type.
        # yapf: disable
        elif self._in_type.is_floating and self._out_type.is_integer:
            if self._params_proto.float_max_to_one:
                if self._out_type == tf.dtypes.uint8:
                    inputs *= 256.0
                    inputs = tf.math.minimum(inputs, 2 ** 8 - 1)
                elif self._out_type == tf.dtypes.uint16:
                    inputs *= (2 ** 16)
                    inputs = tf.math.minimum(inputs, 2 ** 16 - 1)
                elif self._out_type == tf.dtypes.uint32:
                    inputs *= (2 ** 32)
                    inputs = tf.math.minimum(inputs, 2 ** 32 - 1)
                elif self._out_type == tf.dtypes.int8:
                    inputs *= 128.0
                    inputs = tf.math.minimum(inputs,  127)
                    inputs = tf.math.maximum(inputs, -128)
                elif self._out_type == tf.dtypes.int16:
                    inputs *= (2 ** 15)
                    inputs = tf.math.minimum(inputs,  2 ** 15 - 1)
                    inputs = tf.math.minimum(inputs, -2 ** 15)
                elif self._out_type == tf.dtypes.int32:
                    inputs *= (2 ** 15)
                    inputs = tf.math.minimum(inputs,  2 ** 15 - 1)
                    inputs = tf.math.minimum(inputs, -2 ** 15)
                else:
                    raise TypeError("Unsupported types.")

            inputs = tf.cast(inputs, dtype=self._out_type)
        # yapf: enable

        # Converts from an integer type to a floating-point type.
        # yapf: disable
        elif self._in_type.is_integer and self._out_type.is_floating:
            inputs = tf.cast(inputs, dtype=self._out_type)
            if self._params_proto.float_max_to_one:
                if self._in_type == tf.dtypes.uint8:
                    inputs /= 256.0
                elif self._in_type == tf.dtypes.unit16:
                    inputs /= (2 ** 16)
                elif self._in_type == tf.dtypes.unit32:
                    inputs /= (2 ** 32)
                elif self._in_type == tf.dtypes.int8:
                    inputs /= 128.0
                elif self._in_type == tf.dtypes.int16:
                    inputs /= (2 ** 15)
                elif self._in_type == tf.dtypes.int32:
                    inputs /= (2 ** 32)
                else:
                    raise TypeError("Unsupported types.")
        # yapf: enable

        # Converts from an integer type to another integer type.
        elif self._in_type.is_integer and self._out_type.is_integer:
            inputs = tf.cast(inputs, dtype=self._out_type)
        else:
            raise TypeError("Unsupported types.")

        # Performs conversions to a two-tuple structure if the as_supervised
        # option is true.
        if self._params_proto.as_supervised:
            dataset_output = (inputs, labels)
        else:
            dataset_output = inputs

        return dataset_output

    def process(self, dataset):
        """Performs the actual data type conversion.

        Args:
            dataset: a tf.data.Dataset object.

        Returns:
            A tf.data.Dataset object containing data in a converted format.
        """
        assert isinstance(dataset, tf.data.Dataset)

        for data in dataset:
            output = self._convert_type(data)

        if self._params_proto.as_supervised:
            return dataset.map(lambda data, label: self._convert_type(
                (data, label))).cache()
        else:
            return dataset.map(lambda data: self._convert_type(data))


class DictToTupleDatasetOperation(Tf2DatasetOperation):
    """A class for converting a dictionary dataset to a tuple dataset.

    Example Usage;
        - all_values case.
            params_proto = text_format.Parse('''
                [type.googleapi.com/learning.DictToTupleDatasetOperationParams] {
                    value_values: {
                        keys: "b"
                        keys: "d"
                    }
                }
            ''', any_pb2.Any())

            op = tf2_dataset_operation.DictToTupleDatasetOperation(params_proto)
            actual_dataset = op.process(self._input_dataset)

        - dict_inputs case.
            params_proto = text_format.Parse('''
            '''
            [type.googleapi.com/learning.DictToTupleDatasetOperationParams] {
                dict_inputs: {
                    inputs_keys: "b"
                    inputs_keys: "d"
                    targets_key: "e"
                }
            }
            ''', any_pb2.Any())

            op = tf2_dataset_operation.DictToTupleDatasetOperation(params_proto)
            actual_dataset = op.process(self._input_dataset)
    """
    def __init__(self,
                 any_params_proto,
                 params_dict=None,
                 operation_dict=None):
        # yapf: disable
        super(DictToTupleDatasetOperation, self).__init__(
            any_params_proto, params_dict, operation_dict)
        # yapf: enable

        # Unpacks the any proto message.
        assert isinstance(any_params_proto, any_pb2.Any)

        self._class_params = (
            dataset_operation_params_pb2.DictToTupleDatasetOperationParams())
        any_params_proto.Unpack(self._class_params)

    def process(self, dataset):
        """Converts a dataset in a dictionary into a tuple.

        Args:
            dataset: a tf.data.Dataset object.

        Returns:
            A tf.data.Dataset object containing data in a converted format.
        """
        if self._class_params.HasField("all_values"):
            return dataset.map(lambda inputs: tuple(
                inputs[key] for key in self._class_params.all_values.keys))

        elif self._class_params.HasField("dict_inputs"):
            return dataset.map(lambda inputs: ({
                key: inputs[key]
                for key in self._class_params.dict_inputs.inputs_keys
            }, inputs[self._class_params.dict_inputs.targets_key]))
        else:
            raise ValueError("Unsupported input params")


# TODO(chanw.com) Whether this can be done using a CompositeOperation.
class UtteranceDataPreprocessor(Tf2DatasetOperation):
    """A class for pre-processing parsed UtteranceData.

    We assume that the input dataset contains data in a tuple of two elements:
        (acoustic_data_dict, text_data_dict).
    Each dictionary element contains the data and the length in "SEQ_DATA" and
    "SEQ_LEN" keys respectively. In sum, the entire input data format is as
    follows:

        ({"SEQ_DATA": a batch of acoustic sequence data ,
          "SEQ_LEN": a batch of lengths of acoustic sequence data},
         {"SEQ_DATA": a batch of text sequence data
          "SEQ_LEN": a batch of length of text sequence data})

    Note that the above format is the output format of the
    UtteranceDataStringParser in the
    speech/common/utterance_data_string_parser.py module.

    The final output is a tuple of (inputs, targets) where each element is
    as follows:

    inputs{
        "sequence_data": A float32 tensor with the following shape:
         (batch_size, sequence_length, num_channels)
        "sequence_data_length": An int32 tensor with the following shape:
         (batch_size,)
    }

    targets: A string tensor with the following shape:
        (batch_size, label_length)

    Example Usage:

        The following shows an example of performing feature extraction using
        power mel feature and label processing using SentencePieceTextCodec.

        params_proto = text_format.Parse('''
            [type.googleapi.com/learning.UtteranceDataPreprocessorParams] {
                audio_processing_operation: {
                    class_name: "WaveToFeatureTransformTF2"
                    class_params: {
                        [type.googleapi.com/learning.WaveToFeatureTransformParams] {
                            sampling_rate_hz: 16000.0
                            frame_size_sec: 0.025
                            frame_step_sec: 0.010
                            feature_size: 40
                            lower_edge_hz: 125.0
                            upper_edge_hz: 8000.0
                            filter_bank_energy_floor: 0.0
                            filter_bank_type: TF2_FILTER_BANK
                        }
                    }
                }

                label_processing_operation: {
                    class_name: "SentencePieceTextCodec"
                    class_params: {
                        [type.googleapi.com/learning.SentencePieceTextCodecParams]  {
                            model_name: "testdata/model_unigram_256.model"
                            add_bos: True
                            add_eos: True
                            processing_mode: ENCODING
                        }
                    }
                }
            }
        ''', any_pb2.Any())

        op = tf2_dataset_operation.UtteranceDataPreprocessor(params_proto)
        actual_dataset = op.process(input_dataset)
    """
    def __init__(self,
                 any_params_proto,
                 params_dict=None,
                 operation_dict=None):
        self._params_proto = None
        self._unpacked_params_proto = None

        self._audio_processing_op = []
        self._label_processing_op = []

        self.params_proto = any_params_proto

    @property
    def params_proto(self):
        return self._params_proto

    @params_proto.setter
    def params_proto(self, any_params_proto):
        # Unpacks the any proto message.
        assert isinstance(any_params_proto, any_pb2.Any)

        self._unpacked_params_proto = (
            dataset_operation_params_pb2.UtteranceDataPreprocessorParams())
        any_params_proto.Unpack(self._unpacked_params_proto)
        self._params_proto = any_params_proto

        factory = main_factory.Factory()

        self._audio_processing_op = []
        self._label_processing_op = []

        # Creates a list of audio processing operations.
        for params in self._unpacked_params_proto.audio_processing_operation:
            self._audio_processing_op.append(factory.create_operation(params))

        # Creates a list of label processing operations.
        for params in self._unpacked_params_proto.label_processing_operation:
            self._label_processing_op.append(factory.create_operation(params))

    def _feature_processing(self, inputs, audio_processing_op):
        tf.debugging.Assert(
            isinstance(audio_processing_op, operation.AbstractOperation),
            [tf.constant("Incorrect operation type.")])

        outputs = audio_processing_op.process(inputs)

        tf.debugging.Assert({"SEQ_DATA", "SEQ_LEN"} <= outputs.keys(),
                            [tf.constant("Required keys are not found.")])

        # Passes other values from the inputs.
        for key in inputs.keys():
            if key not in ["SEQ_DATA", "SEQ_LEN"]:
                outputs[key] = inputs[key]

        return outputs

    def _label_processing(self, inputs, label_processing_op):
        tf.debugging.Assert(
            isinstance(label_processing_op, operation.AbstractOperation),
            [tf.constant("Incorrect operation type.")])

        # Checks whether it is derived from Operation.
        outputs = label_processing_op.process(inputs)

        tf.debugging.Assert({"SEQ_DATA", "SEQ_LEN"} <= outputs.keys(),
                            [tf.constant("Required keys are not found.")])

        # Passes other values from the inputs.
        for key in inputs.keys():
            if key not in ["SEQ_DATA", "SEQ_LEN"]:
                outputs[key] = inputs[key]

        return outputs

    #@tf.function  TOODO(chanw.com) Check whether we can enable tf.function.
    #
    # Adding tf.function causes the following error.
    #   "No unary variant device copy function found for direction".
    # At this point, we are not sure about the reason.
    def process(self, dataset):
        """Returns a dataset containing processed acoustic data and labels.

        Args:
            dataset: A tf.data.Dataset input to this operation.
            This dataset should a tuple of dict in the following format:

            ({"SEQ_DATA": a batch of acoustic sequence data ,
              "SEQ_LEN": a batch of lengths of acoustic sequence data},
             {"SEQ_DATA": a batch of text sequence data
              "SEQ_LEN": a batch of length of text sequence data})

        Returns:
            A processed tf.data.Dataset containing a tuple of dicts.
                The format is as follows:
                outputs = ("SEQ_DATA": data, "SEQ_LEN": len,
                           "SEQ_DATA": data, "SEQ_LEN": len)

                Note that the outputs[0] is the batch of acoustic data and
                outputs[1] is the corresponding batch of the text data.
        """
        isinstance(dataset, tf.data.Dataset)

        # Performs processing of acoustic data.
        for audio_processing in self._audio_processing_op:
            dataset = dataset.map(
                lambda acoust, text:
                (self._feature_processing(acoust, audio_processing), text),
                num_parallel_calls=tf.data.AUTOTUNE)

        # Performs processing of labels.
        for label_processing in self._label_processing_op:
            dataset = dataset.map(
                lambda acoust, text:
                (acoust, self._label_processing(text, label_processing)),
                num_parallel_calls=tf.data.AUTOTUNE)

        return dataset
