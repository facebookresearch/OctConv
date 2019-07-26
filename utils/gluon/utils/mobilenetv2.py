# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""MobileNetV2, implemented in Gluon. (support OctConv).
code is based on:
https://github.com/apache/incubator-mxnet/tree/master/python/mxnet/gluon
"""

__all__ = ['mobilenet_v2_100',
           'mobilenet_v2_1125',           
           ]

from mxnet import gluon
from mxnet.context import cpu
from mxnet.gluon.nn import BatchNorm
from mxnet.gluon.block import HybridBlock

# import gluon.nn as nn
from . import octconv as nn


class RELU6(nn.HybridBlock):
    """Relu6 used in MobileNetV2."""
    def __init__(self, **kwargs):
        super(RELU6, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.clip(x, 0, 6, name="relu6")

# use RELU6 if gluon.nn not support 'relu6'
_op_act = nn.Activation

class _BottleneckV1(HybridBlock):
    """ResNetV1 BottleneckV1
    """
    # pylint: disable=unused-argument
    def __init__(self, in_planes, mid_planes, out_planes, strides=1,
                 norm_kwargs=None, last_gamma=False, name_prefix=None,
                 **kwargs):
        super(_BottleneckV1, self).__init__(prefix=name_prefix)

        self.use_shortcut = strides == 1 and in_planes == out_planes

        with self.name_scope():
            num_group = sum((c if c > 0 else 0 for c in mid_planes))

            # extract information
            self.conv1 = nn.Conv2D(channels=mid_planes, in_channels=in_planes,
                                  kernel_size=1, use_bias=False, prefix='conv1')
            self.bn1 = nn.BatchNorm(in_channels=mid_planes, prefix='bn1',
                                  **({} if norm_kwargs is None else norm_kwargs))
            self.relu1 = _op_act('relu6')
            # capture spatial relations
            self.conv2 = nn.Conv2D(channels=mid_planes, in_channels=mid_planes,
                                  kernel_size=3, padding=1, groups=num_group,
                                  strides=strides, use_bias=False, prefix='conv2')
            self.bn2 = nn.BatchNorm(in_channels=mid_planes, prefix='bn2',
                                  **({} if norm_kwargs is None else norm_kwargs))
            self.relu2 = _op_act('relu6')
            # embeding back to information highway
            self.conv3 = nn.Conv2D(channels=out_planes, in_channels=mid_planes,
                                  kernel_size=1, use_bias=False, prefix='conv3')
            self.bn3 = nn.BatchNorm(in_channels=out_planes, prefix='bn3',
                                  gamma_initializer='zeros' if (last_gamma and \
                                  self.use_shortcut) else 'ones',
                                  **({} if norm_kwargs is None else norm_kwargs))

    def _sum(self, x1, x2):
        if (x1 is not None) and (x2 is not None):
            return x1 + x2
        else:
            return x1 if x2 is None else x2

    def hybrid_forward(self, F, x1, x2=None):
        x = (x1, x2)
        shortcut = x

        out = self.relu1(*self.bn1(*self.conv1(*x)))
        out = self.relu2(*self.bn2(*self.conv2(*out)))
        out = self.bn3(*self.conv3(*out))

        if self.use_shortcut:
            out = (self._sum(out[0], shortcut[0]), self._sum(out[1], shortcut[1]))

        return out


class _MobileNetV2(HybridBlock):
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

    def __init__(self, multiplier=1.0, classes=1000, ratio=0.,
                 norm_kwargs=None, final_drop=0., last_gamma=False,
                 name_prefix=None, **kwargs):
        super(_MobileNetV2, self).__init__(prefix=name_prefix)
        # reference:
        # - Howard, Andrew G., et al.
        #   "Mobilenets: Efficient convolutional neural networks for mobile vision applications."
        #   arXiv preprint arXiv:1704.04861 (2017).
        in_channels = [int(multiplier * x) for x in
                        [32] + [16] + [24] * 2 + [32] * 3 + [64] * 4 + [96] * 3 + [160] * 3]
        mid_channels = [int(t * x) for t, x in zip([1] + [6] * 16, in_channels)]
        out_channels = [int(multiplier * x) for t, x in zip([1] + [6] * 16,
                        [16] + [24] * 2 + [32] * 3 + [64] * 4 + [96] * 3 + [160] * 3 + [320])]
        strides = [1, 2] * 2 + [1, 1, 2] + [1] * 6 + [2] + [1] * 3
        in_ratios = [0.] + [ratio] * 13 + [0.] * 3
        ratios = [ratio] * 13 + [0.] * 4
        last_channels = int(1280 * multiplier) if multiplier > 1.0 else 1280

        with self.name_scope():
            self.conv1 = gluon.nn.HybridSequential()
            self.conv1.add(gluon.nn.Conv2D(channels=int(32 * multiplier),
                            kernel_size=3, padding=1, strides=2, use_bias=False,
                            prefix='conv1_'))
            self.conv1.add(gluon.nn.BatchNorm(prefix='bn1_',
                            **({} if norm_kwargs is None else norm_kwargs)))
            self.conv1.add(RELU6())
            # ------------------------------------------------------------------
            stage_index, i = 1, 0
            for k, (in_c, mid_c, out_c, s, ir, r) in enumerate(zip(in_channels, mid_channels, out_channels, strides, in_ratios, ratios)):
                stage_index += 1 if s > 1 else 0
                i = 0 if s > 1 else (i + 1)
                name = 'L%d_B%d' % (stage_index, i)
                # -------------------------------------
                in_c = (in_c, -1)
                mid_c = self._get_channles(mid_c, r)
                out_c = (out_c, -1)
                # -------------------------------------
                setattr(self, name, _BottleneckV1(in_c, mid_c, out_c,
                                             strides=s,
                                             norm_kwargs=None,
                                             last_gamma=last_gamma,
                                             name_prefix="%s_" % name))
            # ------------------------------------------------------------------
            self.tail = gluon.nn.HybridSequential()
            self.tail.add(gluon.nn.Conv2D(channels=last_channels, in_channels=out_channels[-1],
                                    kernel_size=1, use_bias=False, prefix='tail-conv_'))
            self.tail.add(gluon.nn.BatchNorm(prefix='tail-bn_',
                                    **({} if norm_kwargs is None else norm_kwargs)))
            self.tail.add(RELU6())
            # ------------------------------------------------------------------
            self.avgpool = gluon.nn.GlobalAvgPool2D()
            self.drop = gluon.nn.Dropout(final_drop) if final_drop > 0. else lambda x: (x)
            self.classifer = gluon.nn.Conv2D(in_channels=last_channels, channels=classes,
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

        x = self.tail(x[0])
        x = self.avgpool(x)
        x = self.drop(x)
        x = self.classifer(x)
        x = self.flat(x)
        return x
           

def mobilenet_v2_100(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), ratio=0., **kwargs):
    net = _MobileNetV2(multiplier=1.0, ratio=ratio,
                       name_prefix='M100_', norm_kwargs=None, **kwargs)
    if pretrained:
        raise NotImplementedError('Please manually load the pretrained params.')
    return net
    
def mobilenet_v2_1125(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), ratio=0., **kwargs):
    net = _MobileNetV2(multiplier=1.125, ratio=ratio,
                       name_prefix='M1125_', norm_kwargs=None, **kwargs)
    if pretrained:
        raise NotImplementedError('Please manually load the pretrained params.')
    return net
