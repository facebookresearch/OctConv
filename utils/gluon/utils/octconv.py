# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Octave Convolution, implemented in Gluon."""
# pylint: disable=arguments-differ,unused-argument,missing-docstring
from __future__ import division

from mxnet.gluon import nn
from mxnet.gluon.nn import *
from mxnet.gluon.block import HybridBlock


__all__ = ['Conv2D',
           'Activation',
           'AvgPool2D',
           'MaxPool2D',
           'BatchNorm',
           'Adapter']


class _upsampling(HybridBlock):
    def __init__(self, scales, sample_type='nearest', **kwargs):
        super(_upsampling, self).__init__(**kwargs)

        assert type(scales) is int or scales[0] == scales[1], \
            "TODO: current upsampling requires all dimensions share the same scale"

        self.scale = scales if type(scales) is int else scales[0]
        self.sample_type = sample_type

    def hybrid_forward(self, F, x):
        return F.UpSampling(x, scale=self.scale, sample_type=self.sample_type)


class Adapter(HybridBlock):
    def __init__(self, **kwargs):
        super(Adapter, self).__init__(**kwargs)

    def hybrid_forward(self, F, x_h, x_l=None):
        if x_l is not None:
            x_l = F.UpSampling(x_l, scale=2, sample_type='nearest')
            x_h = F.Concat(x_h, x_l, dim=1)
        return x_h

class Activation(HybridBlock):
    def __init__(self, activation='relu', **kwargs):
        # options: {'relu', 'sigmoid', 'softrelu', 'softsign', 'tanh'}
        super(Activation, self).__init__(**kwargs)
        self.activation = activation

    def hybrid_forward(self, F, x_h, x_l=None):
        if self.activation == 'relu6':
            func = (lambda x: F.clip(x, 0, 6)) # relu6
        else:
            func = getattr(F, self.activation)
        x_h = func(x_h) if x_h is not None else None
        x_l = func(x_l) if x_l is not None else None
        return (x_h, x_l)


class SE(HybridBlock):
    def __init__(self, channels, in_channels=0, prefix=None, **kwargs):
        super(SE, self).__init__(prefix=prefix, **kwargs)
        # be compatible to conventional convolution
        in_c_h, in_c_l = in_channels if type(in_channels) is tuple else (in_channels, -1)
        assert in_c_l != 0 and in_c_h != 0, \
            "TODO: current version has to specify the `in_channels' to determine the computation graph, but got {}".format(in_channels)

        has_h = in_c_h > 0
        has_l = in_c_l > 0

        with self.name_scope():
            self.conv1 = nn.Conv2D(channels, kernel_size=1, padding=0, prefix='-conv1_')
            self.relu1 = nn.Activation('relu')

            self.conv2_h = nn.Conv2D(in_c_h, kernel_size=1, padding=0,
                                prefix='-conv2-h_') if has_h else lambda x: None
            self.sigmoid2_h = nn.Activation('sigmoid') if has_h else lambda x: None

            self.conv2_l = nn.Conv2D(in_c_l, kernel_size=1, padding=0,
                                prefix='-conv2-l_') if has_l else lambda x: None
            self.sigmoid2_l = nn.Activation('sigmoid') if has_l else lambda x: None

    def _broadcast_mul(self, F, x, w):
        if x is None or w is None:
            assert x is None and w is None, "x is {} but w {}".format(x, w)
            return None
        else:
            return F.broadcast_mul(x, w)

    def _concat(self, F, x1, x2):
        if (x1 is not None) and (x2 is not None):
            return F.Concat(x1, x2, dim=1)
        else:
            return x1 if x2 is None else x2

    def hybrid_forward(self, F, x_h, x_l=None):

        out_h = F.contrib.AdaptiveAvgPooling2D(x_h, output_size=1) if x_h is not None else None
        out_l = F.contrib.AdaptiveAvgPooling2D(x_l, output_size=1) if x_l is not None else None
        out = self._concat(F, out_h, out_l)

        out = self.relu1(self.conv1(out))

        w_h = self.sigmoid2_h(self.conv2_h(out))
        w_l = self.sigmoid2_l(self.conv2_l(out))

        x_h = self._broadcast_mul(F, x_h, w_h)
        x_l = self._broadcast_mul(F, x_l, w_l)

        return (x_h, x_l)


class BatchNorm(HybridBlock):
    def __init__(self, in_channels=0, gamma_initializer='ones', prefix=None, center=True, scale=True, **kwargs):
        super(BatchNorm, self).__init__(prefix=prefix, **kwargs)
        # be compatible to conventional convolution
        in_c_h, in_c_l = in_channels if type(in_channels) is tuple else (in_channels, -1)
        assert in_c_l != 0 and in_c_h != 0, \
            "TODO: current version has to specify the `in_channels' to determine the computation graph, but got {}".format(in_channels)

        with self.name_scope():
            self.bn_h = nn.BatchNorm(gamma_initializer=gamma_initializer, center=center, scale=scale, prefix='-h_', **kwargs) if in_c_h >= 0 else lambda x: (x)
            self.bn_l = nn.BatchNorm(gamma_initializer=gamma_initializer, center=center, scale=scale, prefix='-l_', **kwargs) if in_c_l >= 0 else lambda x: (x)

    def hybrid_forward(self, F, x_h, x_l=None):
        x_h = self.bn_h(x_h) if x_h is not None else None
        x_l = self.bn_l(x_l) if x_l is not None else None
        return (x_h, x_l)


class AvgPool2D(HybridBlock):
    def __init__(self, pool_size, strides, padding=0, in_channels=0, ceil_mode=False, count_include_pad=False, **kwargs):
        super(AvgPool2D, self).__init__(**kwargs)
        # be compatible to conventional convolution
        in_c_h, in_c_l = in_channels if type(in_channels) is tuple else (in_channels, -1)
        assert in_c_l != 0 and in_c_h != 0, \
            "TODO: current version has to specify the `in_channels' to determine the computation graph, but got {}".format(in_channels)

        with self.name_scope():
            self.pool_h = nn.AvgPool2D(pool_size=pool_size, strides=strides, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad) if in_c_h >= 0 else lambda x: (x)
            self.pool_l = nn.AvgPool2D(pool_size=pool_size, strides=strides, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad) if in_c_l >= 0 else lambda x: (x)

    def hybrid_forward(self, F, x_h, x_l=None):
        x_h = self.pool_h(x_h) if x_h is not None else None
        x_l = self.pool_l(x_l) if x_l is not None else None
        return (x_h, x_l)


class MaxPool2D(HybridBlock):
    def __init__(self, pool_size, strides, padding=0, in_channels=0, ceil_mode=False, count_include_pad=False, **kwargs):
        super(MaxPool2D, self).__init__(**kwargs)
        # be compatible to conventional convolution
        in_c_h, in_c_l = in_channels if type(in_channels) is tuple else (in_channels, -1)
        assert in_c_l != 0 and in_c_h != 0, \
            "TODO: current version has to specify the `in_channels' to determine the computation graph, but got {}".format(in_channels)

        with self.name_scope():
            self.pool_h = nn.MaxPool2D(pool_size=pool_size, strides=strides, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad) if in_c_h >= 0 else lambda x: (x)
            self.pool_l = nn.MaxPool2D(pool_size=pool_size, strides=strides, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad) if in_c_l >= 0 else lambda x: (x)

    def hybrid_forward(self, F, x_h, x_l=None):
        x_h = self.pool_h(x_h) if x_h is not None else None
        x_l = self.pool_l(x_l) if x_l is not None else None
        return (x_h, x_l)


class Conv2D(HybridBlock):
    def __init__(self, channels, kernel_size, strides=(1, 1), use_bias=True,
                 in_channels=0, enable_path=((0, 0), (0, 0)), padding=0,
                 groups=1, sample_type='nearest', prefix=None, **kwargs):
        super(Conv2D, self).__init__(prefix=prefix, **kwargs)
        # be compatible to conventional convolution
        (h2l, h2h), (l2l, l2h) = enable_path
        c_h, c_l = channels if type(channels) is tuple else (channels, 0)
        in_c_h, in_c_l = in_channels if type(in_channels) is tuple else (in_channels, -1)

        assert (in_c_h + in_c_l) == groups or ((in_c_h < 0 or in_c_h/groups >= 1) \
                and (in_c_l < 0 or in_c_l/groups >= 1)), \
            "Constains are not satisfied: (%d+%d)==%d, %d/%d>1, %d/%d>1" % ( \
            in_c_h, in_c_l, groups, in_c_h, groups, in_c_l, groups )
        assert in_c_l != 0 and in_c_h != 0, \
            "TODO: current version has to specify the `in_channels' to determine the computation graph"
        assert strides == 1 or strides == 2 or all((s <= 2 for s in strides)), \
            "TODO: current version only support strides({}) <= 2".format(strides)

        is_dw = False
        # computational graph will be automatic or manually defined
        self.enable_l2l = True if l2l != -1 and (in_c_l >= 0 and c_l > 0) else False
        self.enable_l2h = True if l2h != -1 and (in_c_l >= 0 and c_h > 0) else False
        self.enable_h2l = True if h2l != -1 and (in_c_h >= 0 and c_l > 0) else False
        self.enable_h2h = True if h2h != -1 and (in_c_h >= 0 and c_h > 0) else False
        if groups == (in_c_h + in_c_l): # depthwise convolution
            assert c_l == in_c_l and c_h == in_c_h
            self.enable_l2h, self.enable_h2l = False, False
            is_dw = True
        use_bias_l2l, use_bias_h2l = (False, use_bias) if self.enable_h2l else (use_bias, False)
        use_bias_l2h, use_bias_h2h = (False, use_bias) if self.enable_h2h else (use_bias, False)

        # deal with stride with resizing (here, implemented by pooling)
        s = (strides, strides) if type(strides) is int else strides
        do_stride2 = s[0] > 1 or s[1] > 1

        with self.name_scope():
            self.conv_l2l = None if not self.enable_l2l else nn.Conv2D(
                            channels=c_l, kernel_size=kernel_size, strides=1,
                            padding=padding, groups=groups if not is_dw else in_c_l,
                            use_bias=use_bias_l2l, in_channels=in_c_l,
                            prefix='-l2l_', **kwargs)

            self.conv_l2h = None if not self.enable_l2h else nn.Conv2D(
                            channels=c_h, kernel_size=kernel_size, strides=1,
                            padding=padding, groups=groups,
                            use_bias=use_bias_l2h, in_channels=in_c_l,
                            prefix='-l2h_', **kwargs)

            self.conv_h2l = None if not self.enable_h2l else nn.Conv2D(
                            channels=c_l, kernel_size=kernel_size, strides=1,
                            padding=padding, groups=groups,
                            use_bias=use_bias_h2l, in_channels=in_c_h,
                            prefix='-h2l_', **kwargs)

            self.conv_h2h = None if not self.enable_h2h else nn.Conv2D(
                            channels=c_h, kernel_size=kernel_size, strides=1,
                            padding=padding, groups=groups if not is_dw else in_c_h,
                            use_bias=use_bias_h2h, in_channels=in_c_h,
                            prefix='-h2h_', **kwargs)

            self.l2l_down = (lambda x: (x)) if not self.enable_l2l or not do_stride2 else \
                            nn.AvgPool2D(pool_size=strides, strides=strides, \
                                         ceil_mode=True, count_include_pad=False)

            self.l2h_up = (lambda x: (x)) if not self.enable_l2h or do_stride2 else \
                            _upsampling(scales=(2, 2), sample_type=sample_type)

            self.h2h_down = (lambda x: (x)) if not self.enable_h2h or not do_stride2 else \
                            nn.AvgPool2D(pool_size=strides, strides=strides, \
                                         ceil_mode=True, count_include_pad=False)

            self.h2l_down = (lambda x: (x)) if not self.enable_h2l else \
                            nn.AvgPool2D(pool_size=(2*s[0], 2*s[1]), \
                                         strides=(2*s[0], 2*s[1]), \
                                         ceil_mode=True, count_include_pad=False)

    def _sum(self, x1, x2):
        if (x1 is not None) and (x2 is not None):
            return x1 + x2
        else:
            return x1 if x2 is None else x2

    def hybrid_forward(self, F, x_high, x_low=None):

        x_h2h = self.conv_h2h(self.h2h_down(x_high)) if self.enable_h2h else None
        x_h2l = self.conv_h2l(self.h2l_down(x_high)) if self.enable_h2l else None

        x_l2h = self.l2h_up(self.conv_l2h(x_low)) if self.enable_l2h else None
        x_l2l = self.conv_l2l(self.l2l_down(x_low)) if self.enable_l2l else None

        x_h = self._sum(x_l2h, x_h2h)
        x_l = self._sum(x_l2l, x_h2l)
        return (x_h, x_l)
