# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""ResNetV1bs, implemented in Gluon. (support OctConv)
code is based on:
https://github.com/apache/incubator-mxnet/tree/master/python/mxnet/gluon
"""

__all__ = ['resnet50_v1b',
           'resnext50_32x4d_v1b',
           'resnet152_v1e',
           'resnet152_v1f',
           ]

from mxnet import gluon
from mxnet.context import cpu
from mxnet.gluon.nn import BatchNorm
from mxnet.gluon.block import HybridBlock

# import gluon.nn as nn
from . import octconv as nn


class _BottleneckV1(HybridBlock):
    """ResNetV1 BottleneckV1
    """
    # pylint: disable=unused-argument
    def __init__(self, in_planes, mid_planes, out_planes, groups=1, strides=1,
                 norm_kwargs=None, last_gamma=False, name_prefix=None,
                 down_pos=0, use_se=False, se_planes=-1, extra_bn=False,
                 **kwargs):
        super(_BottleneckV1, self).__init__(prefix=name_prefix)
        assert down_pos in [0, 1, 2], \
            "down_pos value({}) is unknown.".format(down_pos)
        strides1 = strides if down_pos == 0 else 1
        strides2 = strides if down_pos == 1 else 1
        strides3 = strides if down_pos == 2 else 1
        with self.name_scope():
            if extra_bn:
                # note: se-block seems not converage well on ResNet152
                self.bn1a = nn.BatchNorm(in_channels=in_planes, prefix='bn1a',
                                  center=False, scale=False,
                                  **({} if norm_kwargs is None else norm_kwargs))
            # extract information
            self.conv1 = nn.Conv2D(channels=mid_planes, in_channels=in_planes,
                                  kernel_size=1, use_bias=False, strides=strides1,
                                  prefix='conv1')
            self.bn1 = nn.BatchNorm(in_channels=mid_planes, prefix='bn1',
                                  **({} if norm_kwargs is None else norm_kwargs))
            self.relu1 = nn.Activation('relu')
            # capture spatial relations
            self.conv2 = nn.Conv2D(channels=mid_planes, in_channels=mid_planes,
                                  kernel_size=3, padding=1, groups=groups,
                                  strides=strides2, use_bias=False, prefix='conv2')
            self.bn2 = nn.BatchNorm(in_channels=mid_planes, prefix='bn2',
                                  **({} if norm_kwargs is None else norm_kwargs))
            self.relu2 = nn.Activation('relu')
            # embeding back to information highway
            self.conv3 = nn.Conv2D(channels=out_planes, in_channels=mid_planes,
                                  kernel_size=1, use_bias=False, strides=strides3,
                                  prefix='conv3')
            self.bn3 = nn.BatchNorm(in_channels=out_planes, prefix='bn3',
                                  gamma_initializer='zeros' if last_gamma else 'ones',
                                  **({} if norm_kwargs is None else norm_kwargs))

            self.se_block = nn.SE(in_channels=out_planes, channels=se_planes,
                                  prefix='se') if use_se else None

            # this relue is added after fusion
            self.relu3 = nn.Activation('relu')

            if strides != 1 or in_planes != out_planes:
                if extra_bn:
                    # note: se-block seems not converage well on ResNet152
                    self.bn4a = nn.BatchNorm(in_channels=in_planes, prefix='bn4a',
                                  center=False, scale=False,
                                  **({} if norm_kwargs is None else norm_kwargs))
                self.conv4 = nn.Conv2D(channels=out_planes, in_channels=in_planes,
                                  kernel_size=1, strides=strides, use_bias=False,
                                  prefix='conv4')
                self.bn4 = nn.BatchNorm(in_channels=out_planes, prefix='bn4',
                                  **({} if norm_kwargs is None else norm_kwargs))

    def _sum(self, x1, x2):
        if (x1 is not None) and (x2 is not None):
            return x1 + x2
        else:
            return x1 if x2 is None else x2

    def hybrid_forward(self, F, x1, x2=None):
        x = (x1, x2)
        shortcut = x
        
        out = self.bn1a(*x) if hasattr(self, 'bn1a') else x
        out = self.relu1(*self.bn1(*self.conv1(*out)))
        out = self.relu2(*self.bn2(*self.conv2(*out)))
        out = self.bn3(*self.conv3(*out))

        out = out if self.se_block is None else self.se_block(*out)

        if hasattr(self, 'conv4'):
            shortcut = self.bn4a(*x) if hasattr(self, 'bn4a') else x
            shortcut = self.bn4(*self.conv4(*shortcut))

        out = (self._sum(out[0], shortcut[0]), self._sum(out[1], shortcut[1]))
        out = self.relu3(*out)
        return out

class _bL_Stem(HybridBlock):
    """Ref:
    Big-little Net: An Efficient Multi-scale Feature Representation
    for Visual And Speech Recognition
    """
    # pylint: disable=unused-argument
    def __init__(self, in_planes, mid_planes, out_planes, strides=1,
                 norm_kwargs=None, last_gamma=False, name_prefix=None,
                 kernel_sizes=(1,3,1), **kwargs):
        super(_bL_Stem, self).__init__(prefix=name_prefix)
        with self.name_scope():
            # extract information
            self.conv1 = gluon.nn.Conv2D(channels=mid_planes, in_channels=in_planes,
                                  kernel_size=kernel_sizes[0],
			          padding=int((kernel_sizes[0]-1)/2),
                                  use_bias=False, prefix='conv1_')
            self.bn1 = gluon.nn.BatchNorm(in_channels=mid_planes, prefix='bn1_',
                                  **({} if norm_kwargs is None else norm_kwargs))
            self.relu1 = gluon.nn.Activation('relu')
            # capture spatial relations
            self.conv2 = gluon.nn.Conv2D(channels=mid_planes, in_channels=mid_planes,
                                  kernel_size=kernel_sizes[1],
                                  padding=int((kernel_sizes[1]-1)/2),
                                  strides=strides, use_bias=False, prefix='conv2_')
            self.bn2 = gluon.nn.BatchNorm(in_channels=mid_planes, prefix='bn2_',
                                  **({} if norm_kwargs is None else norm_kwargs))
            self.relu2 = gluon.nn.Activation('relu')
            # embeding back to information highway
            self.conv3 = gluon.nn.Conv2D(channels=out_planes, in_channels=mid_planes,
                                  kernel_size=kernel_sizes[2],
                                  padding=int((kernel_sizes[2]-1)/2),
                                  use_bias=False, prefix='conv3_')
            self.bn3 = gluon.nn.BatchNorm(in_channels=out_planes, prefix='bn3_',
                                  gamma_initializer='zeros' if last_gamma else 'ones',
                                  **({} if norm_kwargs is None else norm_kwargs))

            # this relue is added after fusion
            self.relu3 = gluon.nn.Activation('relu')

            if strides != 1 or in_planes != out_planes:
                self.conv4 = gluon.nn.Conv2D(channels=out_planes, in_channels=in_planes,
                                  kernel_size=1, strides=strides, use_bias=False,
                                  prefix='conv4_')
                self.bn4 = gluon.nn.BatchNorm(in_channels=out_planes, prefix='bn4_',
                                  **({} if norm_kwargs is None else norm_kwargs))

    def hybrid_forward(self, F, x):
        shortcut = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if hasattr(self, 'conv4'):
            shortcut = self.bn4(self.conv4(x))

        out = out + shortcut
        out = self.relu3(out)
        return out


class _ResNetV1(HybridBlock):
    """ ResNetV1 Model

    Examples:
        - v1: Post Activation Residual Networks, see [1].
              Subsampling is in the first 1x1 conv (i.e., down_pos=0)
        - v1b: Do subsampling at the second 3x3, see [2][4].
        - v1d: v1b + deep stem (i.e., deep_stem=True)
        - v1e (new): v1d, but do subsampling at the second 1x1
               (i.e., down_pos=2)

    Reference:

        [1] He, Kaiming, et al. "Deep residual learning for image recognition."
        Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

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
                 classes=1000, use_se=False, down_pos=0, replace_maxpool=None,
                 norm_kwargs=None, last_gamma=False, deep_stem=False,
                 final_drop=0., use_global_stats=False, extra_bn=False,
                 name_prefix='', **kwargs):
        super(_ResNetV1, self).__init__(prefix=name_prefix)
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        if use_global_stats:
            norm_kwargs['use_global_stats'] = True
        # initialize residual networks
        k = multiplier
        self.use_se = use_se
        self.extra_bn = extra_bn
        self.groups = groups
        self.down_pos=down_pos
        self.last_gamma = last_gamma
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
                n1, s1, s2 = (32, 2, 1) if replace_maxpool is None else (8, 1, 2)
                self.conv1.add(gluon.nn.Conv2D(channels=int(k*n1), kernel_size=3, padding=1, strides=s1,
                                         use_bias=False, prefix='stem_conv1_'))
                self.conv1.add(gluon.nn.BatchNorm(prefix='stem_bn1_',
                                         **({} if norm_kwargs is None else norm_kwargs)))
                self.conv1.add(gluon.nn.Activation('relu'))
                self.conv1.add(gluon.nn.Conv2D(channels=int(k*32), kernel_size=3, padding=1, strides=s2,
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
            if replace_maxpool is None:
                self.maxpool = gluon.nn.MaxPool2D(pool_size=3, strides=2, padding=1)
            elif replace_maxpool == '3x3':
                self.conv2 = gluon.nn.HybridSequential()
                self.conv2.add(gluon.nn.Conv2D(channels=int(k*64), kernel_size=3, padding=1, strides=2,
                                         use_bias=False, prefix='conv2_'))
                self.conv2.add(gluon.nn.BatchNorm(prefix='bn2_',
                                         **({} if norm_kwargs is None else norm_kwargs)))
                self.conv2.add(gluon.nn.Activation('relu'))
            elif replace_maxpool == 'bottleneck-131':
                self.conv2 = _bL_Stem(in_planes=int(k*64), mid_planes=int(k*32), out_planes=int(k*64),
                                         strides=2, norm_kwargs=norm_kwargs, last_gamma=last_gamma,
                                         kernel_sizes=(1,3,1), name_prefix='stem2_')
            else:
                raise NotImplementedError("replace_maxpool = {} is not implemented".format(replace_maxpool))
            # ------------------------------------------------------------------
            # customized convolution starts from this line
            self.inplanes = (int(k*64), -1) # convert to proposed data format
            self._make_layer(1, block, layers[0], int(k*num_out[0]), num_mid[0], ratio[0])
            self._make_layer(2, block, layers[1], int(k*num_out[1]), num_mid[1], ratio[1], strides=2)
            self._make_layer(3, block, layers[2], int(k*num_out[2]), num_mid[2], ratio[2], strides=2)
            self._make_layer(4, block, layers[3], int(k*num_out[3]), num_mid[3], ratio[3], strides=2)
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
                                      last_gamma=self.last_gamma,
                                      use_se=self.use_se,
                                      extra_bn=self.extra_bn,
                                      down_pos=self.down_pos,
                                      se_planes=int(num_out/16),
                                      name_prefix="%s_" % name))
            self.inplanes = out_planes

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x) if hasattr(self, 'conv2') else self.maxpool(x)

        x = (x, None)
        for iy in range(1, 5):
            # assume the max number of blocks is 50 per stage
            for ib in range(0, 50):
                name = 'L%d_B%d' % (iy, ib)
                if hasattr(self, name):
                    x = getattr(self, name)(*x)

        x = self.avgpool(x[0])
        x = self.drop(x)
        x = self.classifer(x)
        x = self.flat(x)
        return x


def resnet50_v1b(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), ratio=0., **kwargs):
    model = _ResNetV1(_BottleneckV1, [3, 4, 6, 3], ratio=[ratio, ratio, ratio, 0.],
                      name_prefix='R50_', down_pos=1, **kwargs)
    if pretrained:
        raise NotImplementedError('Please manually load the pretrained params.')
    return model

def resnext50_32x4d_v1b(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), ratio=0., **kwargs):
    model = _ResNetV1(_BottleneckV1, [3, 4, 6, 3], ratio=[ratio, ratio, ratio, 0.],
                      num_mid=(128, 256, 512, 1024), groups=32,
                      name_prefix='RX50_', down_pos=1,
                      **kwargs)
    if pretrained:
        raise NotImplementedError('Please manually load the pretrained params.')
    return model
    
def resnet152_v1e(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), ratio=0., **kwargs):
    if 'use_se' in kwargs.keys() and kwargs['use_se']:
        kwargs['extra_bn'] = True
    model = _ResNetV1(_BottleneckV1, [3, 8, 36, 3], ratio=[ratio, ratio, ratio, 0.],
                      name_prefix='R152_', deep_stem=True, down_pos=2,
                      **kwargs)
    if pretrained:
        raise NotImplementedError('Please manually load the pretrained params.')
    return model

def resnet152_v1f(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), ratio=0., **kwargs):
    if 'use_se' in kwargs.keys() and kwargs['use_se']:
        kwargs['extra_bn'] = True
    model = _ResNetV1(_BottleneckV1, [3, 8, 36, 3], ratio=[ratio, ratio, ratio, 0.],
                      name_prefix='R152_', deep_stem=True, down_pos=2,
                      replace_maxpool='bottleneck-131',
                      **kwargs)
    if pretrained:
        raise NotImplementedError('Please manually load the pretrained params.')
    return model
