# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=wildcard-import, unused-wildcard-import
"""Models for Octave Convolution paper where most of code is based on:
https://github.com/apache/incubator-mxnet/tree/master/python/mxnet/gluon
"""

from .resnetv1 import *
from .resnetv2 import *
from .densenet import *
from .mobilenetv1 import *
from .mobilenetv2 import *


__all__ = ['get_model', 'get_model_list']

_models = {
    # used in our paper
    'mobilenet_v1_075':     mobilenet_v1_075,
    'mobilenet_v1_100':     mobilenet_v1_100,
    'mobilenet_v2_100':     mobilenet_v2_100,
    'mobilenet_v2_1125':    mobilenet_v2_1125,
    'resnet152_v1e':        resnet152_v1e,
    'resnet152_v1f':        resnet152_v1f,
    # other examples
    'resnet50_v1b':         resnet50_v1b,
    'resnet50_v2b':         resnet50_v2b,
    'resnext50_32x4d_v1b':  resnext50_32x4d_v1b,
    'resnext50_32x4d_v2b':  resnext50_32x4d_v2b,
    'densenet121':          densenet121,
    }


def get_model(name, **kwargs):
    """Returns a pre-defined model by name

    Parameters
    ----------
    name : str
        Name of the model.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    classes : int
        Number of classes for the output layer.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    HybridBlock
        The model.
    """
    name = name.lower()
    if name not in _models:
        err_str = '"%s" is not among the following model list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(_models.keys())))
        raise ValueError(err_str)
    net = _models[name](**kwargs)
    return net


def get_model_list():
    """Get the entire list of model names in model_zoo.

    Returns
    -------
    list of str
        Entire list of model names in model_zoo.

    """
    return _models.keys()
