# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Some useful aliases for vanilla operaters
"""
import mxnet as mx
    
__all__ = ['Convolution',
           'FullyConnected',
           'BatchNorm',
           'Activation',
           'Pooling',
           'ElementWiseSum',
           'Concat',
           'broadcast_mul',
           'expand_dims',
           'BN_ACT_Conv',
           'Conv_BN',
           'Conv_BN_ACT',
           ]
           
           
# -------------
from mxnet.symbol import Convolution, FullyConnected, Activation, Pooling, ElementWiseSum, Concat, broadcast_mul, expand_dims

# we rewrite the BatchNorm to support zero init gamma
def BatchNorm(data, fix_gamma=False, momentum=0.92, eps=0.0001, zero_init_gamma=False, **kwargs):
    if zero_init_gamma and not fix_gamma:
        gamma = mx.sym.Variable("%s_gamma"%name, init=mx.init.Constant(0.))
        bn = mx.symbol.BatchNorm(data=data, fix_gamma=fix_gamma, momentum=momentum, eps=eps, gamma=gamma, **kwargs)
    else:
        bn = mx.symbol.BatchNorm(data=data, fix_gamma=fix_gamma, momentum=momentum, eps=eps, **kwargs)
    return bn
    
    
# -------------
# useful aliases

def BN_ACT(data, act_type='relu', name=None):    
    bn = BatchNorm(data=data, name=('%s_bn' % name))
    act = Activation(data=bn, act_type=act_type, name=('%s_%s' % (name, act_type)))
    return act
    
def BN_ACT_Conv(data, num_filter, kernel, pad=(0, 0), stride=(1,1), name=None, no_bias=True, num_group=1, act_type='relu'):
    b_act = BN_ACT(data=data, name=name)
    conv = Convolution(data=b_act, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, num_group=num_group, no_bias=no_bias, name=('%s_conv' % name))
    return conv

def Conv_BN(data, num_filter, kernel, pad=(0, 0), stride=(1,1), name=None, no_bias=True, num_group=1, zero_init_gamma=False):
    conv = Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, num_group=num_group, no_bias=no_bias, name=('%s_conv' % name))
    bn = BatchNorm(data=conv, zero_init_gamma=zero_init_gamma, name=('%s_bn' % name))
    return bn

def Conv_BN_ACT(data, num_filter, kernel, pad=(0, 0), stride=(1,1), name=None, no_bias=True, num_group=1, act_type='relu'):
    conv = Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, num_group=num_group, no_bias=no_bias, name=('%s_conv' % name))
    b_act = BN_ACT(data=conv, act_type=act_type, name=name)
    return b_act

