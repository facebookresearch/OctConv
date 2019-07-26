# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""ResNetV2bs, implemented in Gluon. (support OctConv)
code is based on:
https://github.com/apache/incubator-mxnet/tree/master/python/mxnet/gluon
"""

__all__ = ['resnet50_v2b',
           'resnext50_32x4d_v2b',
           ]

from mxnet import gluon
from mxnet.context import cpu
from mxnet.gluon.nn import BatchNorm
from mxnet.gluon.block import HybridBlock

# import gluon.nn as nn
from . import octconv as nn


class _BottleneckV2(HybridBlock):
    """ResNetV2 BottleneckV2
    """
    # pylint: disable=unused-argument
    def __init__(self, in_planes, mid_planes, out_planes, groups=1, strides=1,
                 norm_kwargs=None, name_prefix=None,
                 down_pos=0, use_se=False, se_planes=-1, **kwargs):
        super(_BottleneckV2, self).__init__(prefix=name_prefix)
        assert down_pos in [0, 1, 2], \
            "down_pos value({}) is unknown.".format(down_pos)
        strides1 = strides if down_pos == 0 else 1
        strides2 = strides if down_pos == 1 else 1
        strides3 = strides if down_pos == 2 else 1
        with self.name_scope():
            # extract information
            self.bn1 = nn.BatchNorm(in_channels=in_planes, prefix='bn1',
                                  **({} if norm_kwargs is None else norm_kwargs))
            self.relu1 = nn.Activation('relu')
            self.conv1 = nn.Conv2D(channels=mid_planes, in_channels=in_planes,
                                  kernel_size=1, use_bias=False, strides=strides1,
                                  prefix='conv1')
            # capture spatial relations
            self.bn2 = nn.BatchNorm(in_channels=mid_planes, prefix='bn2',
                                  **({} if norm_kwargs is None else norm_kwargs))
            self.relu2 = nn.Activation('relu')
            self.conv2 = nn.Conv2D(channels=mid_planes, in_channels=mid_planes,
                                  kernel_size=3, padding=1, groups=groups,
                                  strides=strides2, use_bias=False, prefix='conv2')
            # embeding back to information highway
            self.bn3 = nn.BatchNorm(in_channels=mid_planes, prefix='bn3',
                                  **({} if norm_kwargs is None else norm_kwargs))
            self.relu3 = nn.Activation('relu')
            self.conv3 = nn.Conv2D(channels=out_planes, in_channels=mid_planes,
                                  kernel_size=1, use_bias=False, strides=strides3,
                                  prefix='conv3')

            self.se_block = nn.SE(in_channels=out_planes, channels=se_planes,
                                  prefix='se') if use_se else None

            if strides != 1 or in_planes != out_planes:
                self.bn4 = nn.BatchNorm(in_channels=in_planes, prefix='bn4',
                                  **({} if norm_kwargs is None else norm_kwargs))
                self.relu4 = nn.Activation('relu')
                self.conv4 = nn.Conv2D(channels=out_planes, in_channels=in_planes,
                                  kernel_size=1, strides=strides,
                                  prefix='conv4')

    def _sum(self, x1, x2):
        if (x1 is not None) and (x2 is not None):
            return x1 + x2
        else:
            return x1 if x2 is None else x2

    def hybrid_forward(self, F, x1, x2=None):
        x = (x1, x2)
        shortcut = x

        out = self.conv1(*self.relu1(*self.bn1(*x)))
        out = self.conv2(*self.relu2(*self.bn2(*out)))
        out = self.conv3(*self.relu3(*self.bn3(*out)))

        out = out if self.se_block is None else self.se_block(*out)

        if hasattr(self, 'conv4'):
            shortcut = self.conv4(*self.relu4(*self.bn4(*x)))

        out = (self._sum(out[0], shortcut[0]), self._sum(out[1], shortcut[1]))
        return out


class _ResNetV2(HybridBlock):
    """ ResNetV2 Model

    Examples:
        - v2: Post Activation Residual Networks, see [1].
              Subsampling is in the first 1x1 conv (i.e., down_pos=0)
        - v2b: Do subsampling at the second 3x3, see [2][4].
        - v2d: v1b + deep stem (i.e., deep_stem=True)

    Reference:

        [1] He, Kaiming, et al. "Identity Mappings in Deep Residual Networks."
        Proceedings of the European Conference on Computer Vision. 2016.

        [2] Xie, Saining, et al. "Aggregated residual transformations for deep neural networks."
        Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017.

        [3] Hu, Jie, Li Shen, and Gang Sun. "Squeeze-and-excitation networks."
        Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.

        [4] https://github.com/facebook/fb.resnet.torch
    """
    # pylint: disable=unused-variable
    def __init__(self, block, layers, groups=1, multiplier=1.,
                 ratio=(0., 0., 0., 0.),
                 num_out=(256, 512, 1024, 2048),
                 num_mid=( 64, 128,  256,  512),
                 classes=1000, use_se=False, down_pos=0,
                 norm_kwargs=None, last_gamma=False, deep_stem=False,
                 final_drop=0., use_global_stats=False,
                 name_prefix='', **kwargs):
        super(_ResNetV2, self).__init__(prefix=name_prefix)
        assert last_gamma == False, "last_gamma should be False for ResNetV2"
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        if use_global_stats:
            norm_kwargs['use_global_stats'] = True
        # initialize residual networks
        k = multiplier
        self.use_se = use_se
        self.groups = groups
        self.down_pos=down_pos
        self.norm_kwargs = norm_kwargs

        with self.name_scope():
            self.conv1 = gluon.nn.HybridSequential()
            if not deep_stem:
                self.conv1.add(gluon.nn.Conv2D(channels=int(k*64), kernel_size=7, padding=3, strides=2,
                                         use_bias=False, prefix='conv1_'))
                self.conv1.add(gluon.nn.BatchNorm(prefix='bn1_',
                                         **({} if norm_kwargs is None else norm_kwargs)))
                self.conv1.add(gluon.nn.Activation('relu'))
            else:
                self.conv1.add(gluon.nn.Conv2D(channels=int(k*32), kernel_size=3, padding=1, strides=2,
                                         use_bias=False, prefix='stem_conv1_'))
                self.conv1.add(gluon.nn.BatchNorm(prefix='stem_bn1_',
                                         **({} if norm_kwargs is None else norm_kwargs)))
                self.conv1.add(gluon.nn.Activation('relu'))
                self.conv1.add(gluon.nn.Conv2D(channels=int(k*32), kernel_size=3, padding=1, strides=1,
                                         use_bias=False, prefix='stem_conv2_'))
                self.conv1.add(gluon.nn.BatchNorm(prefix='stem_bn2_',
                                         **({} if norm_kwargs is None else norm_kwargs)))
                self.conv1.add(gluon.nn.Activation('relu'))
                self.conv1.add(gluon.nn.Conv2D(channels=int(k*64), kernel_size=3, padding=1, strides=1,
                                         use_bias=False, prefix='stem_conv3_'))
                self.conv1.add(gluon.nn.BatchNorm(prefix='stem_bn3_',
                                         **({} if norm_kwargs is None else norm_kwargs)))
                self.conv1.add(gluon.nn.Activation('relu'))
            # ------------------------------------------------------------------
            self.maxpool = gluon.nn.MaxPool2D(pool_size=3, strides=2, padding=1)
            # ------------------------------------------------------------------
            # customized convolution starts from this line
            self.inplanes = (int(k*64), -1) # convert to proposed data format
            self._make_layer(1, block, layers[0], int(k*num_out[0]), num_mid[0], ratio[0])
            self._make_layer(2, block, layers[1], int(k*num_out[1]), num_mid[1], ratio[1], strides=2)
            self._make_layer(3, block, layers[2], int(k*num_out[2]), num_mid[2], ratio[2], strides=2)
            self._make_layer(4, block, layers[3], int(k*num_out[3]), num_mid[3], ratio[3], strides=2)
            # ------------------------------------------------------------------
            self.tail = gluon.nn.HybridSequential()
            self.tail.add(gluon.nn.BatchNorm(prefix='tail-bn_',
                                    **({} if norm_kwargs is None else norm_kwargs)))
            self.tail.add(gluon.nn.Activation('relu'))
            # ------------------------------------------------------------------
            self.avgpool = gluon.nn.GlobalAvgPool2D()
            self.drop = gluon.nn.Dropout(final_drop) if final_drop > 0. else lambda x: (x)
            self.classifer = gluon.nn.Conv2D(in_channels=int(k*num_out[3]), channels=classes,
                                       kernel_size=1, prefix='classifier_')
            self.flat = gluon.nn.Flatten()

    def _make_layer(self, stage_index, block, blocks, num_out, num_mid, ratio=0., strides=1):

        # -1 stands for None
        mid_planes = (num_mid - int(ratio * num_mid), int(ratio * num_mid))
        mid_planes = tuple(c if c != 0 else -1 for c in mid_planes)
        out_planes = (num_out - int(ratio * num_out), int(ratio * num_out))
        out_planes = tuple(c if c != 0 else -1 for c in out_planes)

        for i in range(0, blocks):
            name = 'L%d_B%d' % (stage_index, i)
            setattr(self, name, block(self.inplanes, mid_planes, out_planes,
                                      groups=self.groups,
                                      strides=(strides if i == 0 else 1),
                                      norm_kwargs=self.norm_kwargs,
                                      use_se=self.use_se,
                                      down_pos=self.down_pos,
                                      se_planes=int(num_out/16),
                                      name_prefix="%s_" % name))
            self.inplanes = out_planes

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = (x, None)
        for iy in range(1, 5):
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


def resnet50_v2b(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), ratio=0., **kwargs):
    model = _ResNetV2(_BottleneckV2, [3, 4, 6, 3], ratio=[ratio, ratio, ratio, 0.],
                      name_prefix='R50_', down_pos=1,
                      **kwargs)
    if pretrained:
        raise NotImplementedError('Please manually load the pretrained params.')
    return model

def resnext50_32x4d_v2b(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), ratio=0., **kwargs):
    model = _ResNetV2(_BottleneckV2, [3, 4, 6, 3], ratio=[ratio, ratio, ratio, 0.],
                      num_mid=(128, 256, 512, 1024), groups=32,
                      name_prefix='RX50_', down_pos=1,
                      **kwargs)
    if pretrained:
        raise NotImplementedError('Please manually load the pretrained params.')
    return model
