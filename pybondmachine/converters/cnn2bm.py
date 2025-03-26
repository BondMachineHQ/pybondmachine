from typing import Union
import logging

import numpy as np
from keras import layers as klayers, models as kmodels

logger = logging.getLogger(__name__)

def _serialize_conv_nd_layer(layer: Union[klayers.Conv1D, klayers.Conv2D]):
    if layer.data_format.lower() != "channels_last":
        raise Exception(
        f"""Unsupported data_format value '{layer.data_format}' of layer '{layer.name}'.
            Only 'channels_last' data format is currently supported"""
        )

    if layer.padding.lower() != "valid":
        raise Exception(
        f"""Unsupported padding value '{layer.padding}' of layer '{layer.name}'
            Only 'valid' padding is currently supported"""
        )

    filters = layer.filters
    kernel_size = layer.kernel_size
    conv_dims = len(kernel_size)

    if len(layer.strides) != conv_dims or list(layer.strides).count(1) != conv_dims:
        raise Exception(
        f"""Unsupported stride value '{layer.strides}' of layer '{layer.name}'.
            Only a stride of 1 across all dimensions is currently supported"""
        )

    raw_weights, raw_biases = layer.get_weights()
    assert raw_weights.shape[:conv_dims] == kernel_size and raw_weights.shape[-1] == filters
    biases = raw_biases.astype(np.float64, casting="safe")

    channels = raw_weights.shape[-2]
    elements_per_filter = np.prod(kernel_size)
    # Keras stores the weights with ordering (kernelSize, channels, kernels).
    # Instead what we'd like to have is (kernels, channels, kernelSize).
    # The following lines perform this transformation.
    weights = (raw_weights
        .astype(np.float64, casting="safe")
        # (kernelSize, channels, kernels) => (kernelElements, kernels)
        .reshape((elements_per_filter * channels, filters))
        # (kernelElements, kernels) => (kernels, kernelElements)
        .transpose())

    filter_weights = [{ "weights": filter_ws.tolist(), "bias": filter_b } for filter_ws, filter_b in zip(weights, biases)]

    return {
        "type": f"Conv{conv_dims}D",
        "activation": layer.activation.__name__,
        "filtersShape": kernel_size,
        "filters": filter_weights,
    }

def _serialize_max_pool_nd_layer(layer: Union[klayers.MaxPool1D, klayers.MaxPool2D]):
    if layer.data_format.lower() != "channels_last":
        raise Exception(
        f"""Unsupported data_format value '{layer.data_format}' of layer '{layer.name}'.
            Only 'channels_last' data format is currently supported"""
        )

    if layer.padding.lower() != "valid":
        raise Exception(
        f"""Unsupported padding value '{layer.padding}' of layer '{layer.name}'
            Only 'valid' padding is currently supported"""
        )

    dims = None
    match type(layer):
        case klayers.MaxPool1D:
            dims = 1
        case klayers.MaxPool2D:
            dims = 2
    assert dims != None

    return {
        "type": f"Pool{dims}D",
        "operation": "max",
        "poolSize": layer.pool_size,
        "stride": layer.strides,
    }

def _serialize_flatten_layer(layer: klayers.Flatten):
    if layer.data_format.lower() != "channels_last":
        raise Exception(
        f"""Unsupported data_format value '{layer.data_format}' of layer '{layer.name}'.
            Only 'channels_last' data format is currently supported"""
        )

    return { "type": "Flatten" }

def _serialize_dense_layer(layer: klayers.Dense):
    if not layer.use_bias:
        raise Exception(
        f"""Unsupported use_bias value '{layer.use_bias}' of layer '{layer.name}'.
            'use_bias' must be enabled"""
        )

    raw_weights, raw_biases = layer.get_weights()
    # Transform weights ordering: (inputNodes, outputNodes) => (outputNodes, inputNodes)
    weights = raw_weights.astype(np.float64, casting="safe").transpose()
    biases = raw_biases.astype(np.float64, casting="safe")

    node_weights = [{ "weights": node_ws.tolist(), "bias": node_b } for node_ws, node_b in zip(weights, biases)]

    return {
        "type": "Dense",
        "activation": layer.activation.__name__,
        "outputNodes": node_weights,
    }

def _serialize_layer(layer: klayers.Layer):
    match type(layer):
        case klayers.Conv1D | klayers.Conv2D:
            return _serialize_conv_nd_layer(layer)
        case klayers.MaxPool1D | klayers.MaxPool2D:
            return _serialize_max_pool_nd_layer(layer)
        case klayers.Flatten:
            return _serialize_flatten_layer(layer)
        case klayers.Dense:
            return _serialize_dense_layer(layer)
        case _:
            raise Exception(f"Unrecognized layer '{layer.name}', of type '{type(layer)}'")

def serialize_model(model: kmodels.Sequential):
    if len(model.layers) == 0:
        logger.warning("Empty model provided")
        return []

    input_batches, *input_shape = model.layers[0].input.shape
    if input_batches != None:
        raise Exception("Batches are not supported yet")

    serialized_layers = list(map(_serialize_layer, model.layers))
    return {
        # TODO: 'batchSize', 'dataType', ...
        "inputShape": input_shape,
        "layers": serialized_layers,
    }

def serialize_single_layer(layer: klayers.Layer):
    return {
        # Removing batch size from input's shape
        "inputShape": layer.input.shape[1:],
        "layers": [ _serialize_layer(layer) ],
    }
