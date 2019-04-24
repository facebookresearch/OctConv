# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Implementation of: Octave Convolution
"""
import mxnet as mx

__all__ = ['Convolution',
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


def _gather(operator, srcs, name, **kwargs):
    if type(srcs[0]) is not tuple:
        return operator(*srcs, **kwargs)
    output = []
    appxs = {1: ['-h'], 2: ['-h', '-l']}[len(srcs[0])]
    for i in range(len(srcs[0])):
        data = [src[i] for src in srcs if src[i] is not None]
        output.append(None if len(data) == 0 
                      else data[0] if len(data) == 1 
                      else operator(*data, name=(name+appxs[i]),**kwargs) )
    return tuple(output)
    
def _adopt(operator, srcs, name, **kwargs):
    srcs = srcs if type(srcs) is tuple else srcs
    appxs = {1: ['-h'], 2: ['-h', '-l']}[len(srcs)]
    dst = tuple(operator(src, name=(name+appx), **kwargs) if src is not None else src \
            for src, appx in zip(srcs,appxs))
    dst = dst[0] if len(dst) == 1 else dst
    return dst

# -------------
# we rewrite the BatchNorm to support `zero init gamma'
def _BatchNorm(data, fix_gamma=False, zero_init_gamma=False, name=None, **kwargs):
    assert not (zero_init_gamma and fix_gamma), "setting is illegal."
    if zero_init_gamma and not fix_gamma:
        gamma = mx.sym.Variable("%s_gamma"%name, init=mx.init.Constant(0.))
        bn = mx.sym.BatchNorm(data=data, fix_gamma=fix_gamma, name=name, gamma=gamma, **kwargs)
    else:
        bn = mx.sym.BatchNorm(data=data, fix_gamma=fix_gamma, name=name, **kwargs)
    return bn

# -------------
# main body

def BatchNorm(data, fix_gamma=False, momentum=0.92, eps=0.0001, zero_init_gamma=False, name=None):
    return _adopt(_BatchNorm, data, name,
                fix_gamma=fix_gamma, momentum=momentum, eps=eps, zero_init_gamma=zero_init_gamma)

def Activation(data, act_type='relu', name=None):
    return _adopt(mx.sym.Activation, data, name, act_type=act_type)

def Pooling(data, name, **kwargs):
    return _adopt(mx.sym.Pooling, data, name, **kwargs)
    
def expand_dims(data, name, **kwargs):
    return _adopt(mx.sym.expand_dims, data, name, **kwargs)
    
def ElementWiseSum(*args, name):
    return _gather(mx.sym.ElementWiseSum, args, name)

def Concat(*args, name, **kwargs):
    return _gather(mx.sym.Concat, args, name, **kwargs)

def broadcast_mul(*args, name, **kwargs):
    assert len(args) == 2
    return _gather(mx.sym.broadcast_mul, args, name, **kwargs)

def Convolution(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), num_group=1, no_bias=True, name=None):
    data_h, data_l = data if type(data) is tuple else (data, None)
    num_high, num_low = num_filter if type(num_filter) is tuple else (num_filter, 0)

    assert num_high >= 0 and num_low >= 0 
    assert stride == (1, 1) or stride == (2, 2), "stride = {} is not supported yet".format(stride)

    data_h2l, data_h2h, data_l2l, data_l2h = None, None, None, None
    depthwise = True if num_filter == num_group else False
    
    '''processing high frequency group'''
    if data_h is not None:
        # High -> High
        data_h = mx.sym.Pooling(data=data_h, pool_type="avg", kernel=(2, 2), pad=(0, 0), stride=(2, 2)) if stride == (2, 2) else data_h
        data_h2h = mx.sym.Convolution(data=data_h, num_filter=num_high, kernel=kernel, stride=(1, 1), pad=pad, num_group=min(num_high, num_group), no_bias=no_bias, name=('%s-h2h' % name)) if num_high > 0 else None
        # High -> Low
        if not depthwise:
            data_h2l = mx.sym.Pooling(data=data_h, pool_type="avg", kernel=(2, 2), pad=(0, 0), stride=(2, 2)) if (num_low > 0) else data_h
            data_h2l = mx.sym.Convolution(data=data_h2l, num_filter=num_low, kernel=kernel, stride=(1, 1), pad=pad, num_group=min(num_low, num_group), no_bias=no_bias, name=('%s-h2l' % name)) if num_low > 0 else None

    '''processing low frequency group'''
    if data_l is not None:
        # Low -> Low
        data_l2l = mx.sym.Pooling(data=data_l, pool_type="avg", kernel=(2, 2), pad=(0, 0), stride=(2, 2)) if (num_low > 0 and stride == (2, 2)) else data_l
        data_l2l = mx.sym.Convolution(data=data_l2l, num_filter=num_low, kernel=kernel, stride=(1, 1), pad=pad, num_group=min(num_low, num_group), no_bias=True, name=('%s-l2l' % name)) if num_low > 0 else None
        # Low -> High
        if not depthwise:
            data_l2h = mx.sym.Convolution(data=data_l, num_filter=num_high, kernel=kernel, stride=(1, 1), pad=pad, num_group=min(num_high, num_group), no_bias=True, name=('%s-l2h' % name)) if num_high > 0 else None
            data_l2h = mx.sym.UpSampling(data_l2h, scale=2, sample_type="nearest", num_args=1) if (num_high > 0 and stride == (1, 1)) else data_l2h

    '''you can force to disable the interaction paths'''
    # data_h2l = None if (data_h2h is not None) and (data_l2l is not None) else data_h2l
    # data_l2h = None if (data_h2h is not None) and (data_l2l is not None) else data_l2h

    output = ElementWiseSum(*[(data_h2h, data_h2l), (data_l2h, data_l2l)], name=name)
    
    # squeeze output (to be backward compatible)
    return output[0] if output[1] is None else output


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


# -------------
# some dirty code [todo]
def FullyConnected(data, num_hidden, no_bias=True, name=None):
    data_h, data_l = data if type(data) is tuple else (data, None)
    num_high, num_low = num_hidden if type(num_hidden) is tuple else (num_hidden, 0)

    assert num_high >= 0 and num_low >= 0 
    data_h2l, data_h2h, data_l2l, data_l2h = None, None, None, None
    
    if data_h is not None:
        data_h2h = mx.sym.FullyConnected(data=data_h, num_hidden=num_high, no_bias=no_bias, name=('%s-h2h' % name)) if num_high > 0 else None
        data_h2l = mx.sym.FullyConnected(data=data_h, num_hidden=num_low, no_bias=no_bias, name=('%s-h2l' % name)) if num_low > 0 else None

    if data_l is not None:
        data_l2h = mx.sym.FullyConnected(data=data_l, num_hidden=num_high, no_bias=True, name=('%s-l2h' % name)) if num_high > 0 else None
        data_l2l = mx.sym.FullyConnected(data=data_l, num_hidden=num_low, no_bias=True, name=('%s-l2l' % name)) if num_low > 0 else None

    output = ElementWiseSum(*[(data_h2h, data_h2l), (data_l2h, data_l2l)], name=name)
    
    # squeeze output (to be backward compatible)
    return output[0] if output[1] is None else output
