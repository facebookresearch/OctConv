# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""DenseNet, implemented in Gluon. (support OctConv)
code is based on:
https://github.com/apache/incubator-mxnet/tree/master/python/mxnet/gluon
"""

__all__ = ['densenet121',
          ]

from mxnet import gluon
from mxnet.context import cpu
from mxnet.gluon.nn import BatchNorm
from mxnet.gluon.block import HybridBlock
from mxnet.gluon.contrib.nn import HybridConcurrent, Identity

# import gluon.nn as nn
from . import octconv as nn


class _DenseBlock(HybridBlock):
    def __init__(self, in_channels, out_channels, bn_size=4,
                 norm_kwargs=None, name_prefix=None):
        super(_DenseBlock, self).__init__(prefix=name_prefix)

        num_c1 = (bn_size * out_channels[0], bn_size * out_channels[1])
        num_c1 = tuple(int(c) if c > 0 else -1 for c in num_c1)

        with self.name_scope():
            # 1x1
            self.bn1 = nn.BatchNorm(in_channels=in_channels, prefix='bn1',
                            **({} if norm_kwargs is None else norm_kwargs))
            self.relu1 = nn.Activation('relu')
            self.conv1 = nn.Conv2D(channels=num_c1, in_channels=in_channels,
                            kernel_size=1, padding=0,
                            use_bias=False, prefix='conv1')
            # 3x3
            self.bn2 = nn.BatchNorm(in_channels=num_c1, prefix='bn2',
                            **({} if norm_kwargs is None else norm_kwargs))
            self.relu2 = nn.Activation('relu')
            self.conv2 = nn.Conv2D(channels=out_channels, in_channels=num_c1,
                            kernel_size=3, padding=1,
                            use_bias=False, prefix='conv2')

    def _concat(self, F, x1, x2):
        if (x1 is not None) and (x2 is not None):
            return F.Concat(x1, x2, dim=1)
        else:
            return x1 if x2 is None else x2

    def hybrid_forward(self, F, x1, x2=None):
        x = (x1, x2)
        out = self.conv1(*self.relu1(*self.bn1(*x)))
        out = self.conv2(*self.relu2(*self.bn2(*out)))

        x = (self._concat(F, x[0], out[0]), self._concat(F, x[1], out[1]))
        return x


class _Transition(HybridBlock):
    def __init__(self, in_channels, out_channels,
                 norm_kwargs=None, name_prefix=None):
        super(_Transition, self).__init__(prefix=name_prefix)
        with self.name_scope():
            self.bn = nn.BatchNorm(in_channels=in_channels, prefix='bn',
                            **({} if norm_kwargs is None else norm_kwargs))
            self.relu = nn.Activation('relu')
            self.conv = nn.Conv2D(channels=out_channels, prefix='conv',
                            in_channels=in_channels, kernel_size=1,
                            padding=0, use_bias=False)
            self.pool = nn.AvgPool2D(in_channels=out_channels, 
                            pool_size=2, strides=2)

    def hybrid_forward(self, F, x1, x2=None):
        x = (x1, x2)
        x = self.conv(*self.relu(*self.bn(*x)))
        x = self.pool(*x)
        return x


# Net
class DenseNet(HybridBlock):
    def __init__(self, num_init_features, growth_rate, block_config,
                 classes=1000, final_drop=0., ratio=(0., 0., 0., 0.),
                 norm_kwargs=None, name_prefix=None, **kwargs):
        super(DenseNet, self).__init__(prefix=name_prefix)
        with self.name_scope():
            in_plane = self._get_channles(num_init_features, ratio[0])
            self.conv1 = nn.Conv2D(in_channels=(3, -1), channels=in_plane,
                                     kernel_size=7, padding=3, strides=2,
                                     use_bias=False, prefix='conv1')
            self.bn1 = nn.BatchNorm(in_channels=in_plane, prefix='bn1',
                                     **({} if norm_kwargs is None else norm_kwargs))
            self.relu1 = nn.Activation('relu')
            # ------------------------------------------------------------------
            self.maxpool = nn.MaxPool2D(in_channels=in_plane,
                                     pool_size=3, strides=2, padding=1)
            # ------------------------------------------------------------------
            # Add dense blocks
            in_plane = num_init_features
            for i, num_blocks in enumerate(block_config):
                stage_index = i + 1
                block_index = 0
                for j in range(num_blocks):
                    # change dimension
                    if j == 0 and i > 0:
                        name = 'L%d_B%d' % (stage_index, block_index)
                        in_plane_t = self._get_channles(in_plane, ratio[i-1])
                        out_plane_t = self._get_channles(int(in_plane/2), ratio[i])
                        setattr(self, name, _Transition(in_channels=in_plane_t,
                                                        out_channels=out_plane_t,
                                                        norm_kwargs=norm_kwargs,
                                                        name_prefix="%s_" % name))
                        block_index += 1
                        in_plane = int(in_plane/2)
                    # main part
                    name = 'L%d_B%d' % (stage_index, block_index)
                    in_plane_t = self._get_channles(in_plane, ratio[i])
                    out_plane_t = self._get_channles(growth_rate, ratio[i])
                    setattr(self, name, _DenseBlock(in_channels=in_plane_t,
                                                    out_channels=out_plane_t,
                                                    norm_kwargs=norm_kwargs,
                                                    name_prefix="%s_" % name))
                    block_index += 1
                    in_plane += growth_rate
            # ------------------------------------------------------------------
            self.tail = gluon.nn.HybridSequential()
            self.tail.add(gluon.nn.BatchNorm(prefix='tail-bn_',
                                    **({} if norm_kwargs is None else norm_kwargs)))
            self.tail.add(gluon.nn.Activation('relu'))
            # ------------------------------------------------------------------
            self.avgpool = gluon.nn.GlobalAvgPool2D()
            self.drop = gluon.nn.Dropout(final_drop) if final_drop > 0. else lambda x: (x)
            self.classifer = gluon.nn.Conv2D(channels=classes,
                                       kernel_size=1, prefix='classifier_')
            self.flat = gluon.nn.Flatten()

    def _get_channles(self, width, ratio):
        width = (width - int(ratio * width), int(ratio * width))
        width = tuple(c if c != 0 else -1 for c in width)
        return width

    def hybrid_forward(self, F, x):
        x = self.relu1(*self.bn1(*self.conv1(x)))
        x = self.maxpool(*x)

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

# Constructor
def get_densenet(num_layers, pretrained=False, ctx=cpu(),
                 ratio=0.,
                 root='~/.mxnet/models', **kwargs):
    r"""Densenet-BC model from the
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_ paper.

    Parameters
    ----------
    num_layers : int
        Number of layers for the variant of densenet. Options are 121, 161, 169, 201.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    """
    # Specification
    densenet_spec = {121: (64, 32, [6, 12, 24, 16]),
                     161: (96, 48, [6, 12, 36, 24]),
                     169: (64, 32, [6, 12, 32, 32]),
                     201: (64, 32, [6, 12, 48, 32])}

    num_init_features, growth_rate, block_config = densenet_spec[num_layers]
    net = DenseNet(num_init_features=num_init_features,
                   growth_rate=growth_rate,
                   block_config=block_config,
                   ratio=(ratio, ratio, ratio, 0.),
                   **kwargs)
    if pretrained:
        raise NotImplementedError('Please manually load the pretrained params.')
    return net

def densenet121(**kwargs):
    return get_densenet(121, name_prefix='D121_', **kwargs)

def densenet161(**kwargs):
    return get_densenet(161, name_prefix='D161_', **kwargs)

def densenet169(**kwargs):
    return get_densenet(169, name_prefix='D169_', **kwargs)

def densenet201(**kwargs):
    return get_densenet(201, name_prefix='D201_', **kwargs)
