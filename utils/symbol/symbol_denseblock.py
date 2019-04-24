# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import symbol_basic
import symbol_octconv

__all__ = ['Connector',
           'DenseBlock',
           ]


def Connector(data, num_out, name, ratio=-1):

    sym = symbol_basic
    if ratio >= 0:
        num_out = (num_out - int(ratio*num_out), int(ratio*num_out))
        sym = symbol_octconv
        
    out = sym.BN_ACT_Conv(data=data, num_filter=num_out, kernel=(1, 1), stride=(1, 1), name=('%s_conv' % name), no_bias=True)
    out = sym.Pooling(data=out, pool_type="avg",  kernel=(2, 2), stride=(2, 2), name=("%s_pool" % name))
    
    return out


def DenseBlock(data, num_mid, num_out, name, ratio=-1):

    sym = symbol_basic
    if ratio >= 0:
        num_mid = (num_mid - int(ratio*num_mid), int(ratio*num_mid))
        num_out = (num_out - int(ratio*num_out), int(ratio*num_out))
        sym = symbol_octconv
        
    out = sym.BN_ACT_Conv(data=data, num_filter=num_mid, kernel=(1, 1), pad=(0, 0), name=('%s_conv1' % name))
    out = sym.BN_ACT_Conv(data=out,  num_filter=num_out, kernel=(3, 3), pad=(1, 1), name=('%s_conv2' % name))

    out = sym.Concat(*[data, out], name=('%s_concat' % name))
    return out

