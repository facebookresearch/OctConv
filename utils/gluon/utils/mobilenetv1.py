# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""MobileNet, implemented in Gluon. (support OctConv)
code is based on:
https://github.com/apache/incubator-mxnet/tree/master/python/mxnet/gluon
"""

__all__ = ['mobilenet_v1_075',
           'mobilenet_v1_100',
           ]

from mxnet import gluon
from mxnet.context import cpu
from mxnet.gluon.nn import BatchNorm
from mxnet.gluon.block import HybridBlock

# import gluon.nn as nn
from . import octconv as nn


class _DWSConv(HybridBlock):
    r"""
    Original Depthwise Separable Convolution
    (-> depthwise convolution -> pointwise convolution)
    """
    def __init__(self, in_channels, channels, stride,
                 norm_kwargs=None, name_prefix=None):
        super(_DWSConv, self).__init__(prefix=name_prefix)
        with self.name_scope():
            # depthwise conv
            self.conv1 = nn.Conv2D(channels=in_channels, in_channels=in_channels,
                            kernel_size=3, padding=1, groups=sum(in_channels),
                            strides=stride, use_bias=False, prefix='conv1')
            self.bn1 = nn.BatchNorm(in_channels=in_channels, prefix='bn1',
                            **({} if norm_kwargs is None else norm_kwargs))
            self.relu1 = nn.Activation('relu')
            # pointwise conv
            self.conv2 = nn.Conv2D(channels=channels, in_channels=in_channels,
                            kernel_size=1, use_bias=False, prefix='conv2')
            self.bn2 = nn.BatchNorm(in_channels=channels, prefix='bn2',
                            **({} if norm_kwargs is None else norm_kwargs))
            self.relu2 = nn.Activation('relu')

    def hybrid_forward(self, F, x1, x2=None):
        x = (x1, x2)
        out = self.relu1(*self.bn1(*self.conv1(*x)))
        out = self.relu2(*self.bn2(*self.conv2(*out)))
        return out


class _MobileNetV1(HybridBlock):
    r"""MobileNet model from the
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861>`_ paper.

    Parameters
    ----------
    multiplier : float, default 1.0
        The width multiplier for controlling the model size. Only multipliers that are no
        less than 0.25 are supported. The actual number of channels is equal to the original
        channel size multiplied by this multiplier.
    classes : int, default 1000
        Number of classes for the output layer.
    """

    def __init__(self, multiplier=1.0, classes=1000,
                 ratio=0.,
                 norm_kwargs=None, final_drop=0.,
                 name_prefix=None, **kwargs):
        super(_MobileNetV1, self).__init__(prefix=name_prefix)
        # reference:
        # - Howard, Andrew G., et al.
        #   "Mobilenets: Efficient convolutional neural networks for mobile vision applications."
        #   arXiv preprint arXiv:1704.04861 (2017).
        dw_channels = [int(x * multiplier) for x in
                       [32, 64] + [128] * 2 + [256] * 2 + [512] * 6 + [1024]]
        channels = [int(x * multiplier) for x in
                       [64] + [128] * 2 + [256] * 2 + [512] * 6 + [1024] * 2]
        strides = [1, 2] * 3 + [1] * 5 + [2, 1]
        dw_ratios = [0.] + [ratio] * 10 + [0.] * 2
        pw_ratios = [ratio] * 10 + [0.] * 3

        with self.name_scope():
            self.conv1 = gluon.nn.HybridSequential()
            self.conv1.add(gluon.nn.Conv2D(channels=int(32 * multiplier),
                            kernel_size=3, padding=1, strides=2, use_bias=False,
                            prefix='conv1_'))
            self.conv1.add(gluon.nn.BatchNorm(prefix='bn1_',
                            **({} if norm_kwargs is None else norm_kwargs)))
            self.conv1.add(gluon.nn.Activation('relu'))

            stage_index, i = 1, 0
            for dwc, pwc, s, dr, pr in zip(dw_channels, channels, strides, dw_ratios, pw_ratios):
                stage_index += 1 if s > 1 else 0
                i = 0 if s > 1 else (i + 1)
                # -------------------------------------
                dwc = self._get_channles(dwc, dr)
                pwc = self._get_channles(pwc, pr)
                # -------------------------------------
                name = 'L%d_B%d' % (stage_index, i)
                setattr(self, name, _DWSConv(in_channels=dwc,
                                             channels=pwc, stride=s,
                                             norm_kwargs=None,
                                             name_prefix="%s_"%name))

            self.drop = gluon.nn.Dropout(final_drop) if final_drop > 0. else lambda x: (x)
            self.classifer = gluon.nn.Conv2D(in_channels=channels[-1], channels=classes,
                                       kernel_size=1, prefix='classifier_')
            self.flat = gluon.nn.Flatten()

    def _get_channles(self, width, ratio):
        width = (width - int(ratio * width), int(ratio * width))
        width = tuple(c if c != 0 else -1 for c in width)
        return width

    def _concat(self, F, x1, x2):
        if (x1 is not None) and (x2 is not None):
            return F.Concat(x1, x2, dim=1)
        else:
            return x1 if x2 is None else x2

    def hybrid_forward(self, F, x):

        x = self.conv1(x)

        x = (x, None)
        for iy in range(1, 10):
            # assume the max number of blocks is 50 per stage
            for ib in range(0, 50):
                name = 'L%d_B%d' % (iy, ib)
                if hasattr(self, name):
                    x = getattr(self, name)(*x)

        x_h, x_l = x
        x_h = F.contrib.AdaptiveAvgPooling2D(x_h, output_size=1) if x_h is not None else None
        x_l = F.contrib.AdaptiveAvgPooling2D(x_l, output_size=1) if x_l is not None else None
        x = self._concat(F, x_h, x_l)

        x = self.drop(x)
        x = self.classifer(x)
        x = self.flat(x)
        return x


def mobilenet_v1_075(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), ratio=0., **kwargs):
    net = _MobileNetV1(multiplier=0.75, ratio=ratio,
                       name_prefix='M075_', norm_kwargs=None, **kwargs)
    if pretrained:
        raise NotImplementedError('Please manually load the pretrained params.')
    return net

def mobilenet_v1_100(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), ratio=0., **kwargs):
    net = _MobileNetV1(multiplier=1.0, ratio=ratio,
                       name_prefix='M100_', norm_kwargs=None, **kwargs)
    if pretrained:
        raise NotImplementedError('Please manually load the pretrained params.')
    return net
