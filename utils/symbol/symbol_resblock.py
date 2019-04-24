# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import symbol_basic
import symbol_octconv
from symbol_seblock import SE_Block

__all__ = ['BottleNeckV1',
           'BottleNeckV2',
           ]


'''Post-activation, see: ResNet (CVPR ver.) for more details'''
def BottleNeckV1(data, num_in, num_mid, num_out, name, first_block=False, stride=(1, 1), num_group=1, use_se=False, zero_init_gamma=False, ratio=-1):
    
    sym = symbol_basic
    if ratio >= 0:
        num_mid = (num_mid - int(ratio*num_mid), int(ratio*num_mid))
        num_out = (num_out - int(ratio*num_out), int(ratio*num_out))
        sym = symbol_octconv # overwrite operators
    
    out = sym.Conv_BN_ACT(data=data, num_filter=num_mid, kernel=(1, 1), pad=(0, 0), name=('%s_conv1' % name))
    out = sym.Conv_BN_ACT(data=out,  num_filter=num_mid, kernel=(3, 3), pad=(1, 1), name=('%s_conv2' % name), stride=stride, num_group=num_group)
    out = sym.Conv_BN(    data=out,  num_filter=num_out, kernel=(1, 1), pad=(0, 0), name=('%s_conv3' % name), zero_init_gamma=zero_init_gamma)
    
    # optional
    out = SE_Block(sym=sym, data=out, num_out=num_out, name=('%s_se' % name)) if use_se else out
    
    if first_block:
        data = sym.Conv_BN(data=data, num_filter=num_out, kernel=(1, 1), pad=(0, 0), name=('%s_conv4' % name), stride=stride)
    
    out = sym.ElementWiseSum(*[data, out], name=('%s_sum' % name))
    out = sym.Activation(data=out, act_type='relu', name=('%s_relu' % name))
    return out
    
    
'''Pre-activation, see: ResNet (ECCV ver.) for more details'''
def BottleNeckV2(data, num_in, num_mid, num_out, name, first_block=False, stride=(1, 1), num_group=1, use_se=False, ratio=-1):
    
    sym = symbol_basic
    if ratio >= 0:
        num_mid = (num_mid - int(ratio*num_mid), int(ratio*num_mid))
        num_out = (num_out - int(ratio*num_out), int(ratio*num_out))
        sym = symbol_octconv # overwrite operators

    out = sym.BN_ACT_Conv(data=data, num_filter=num_mid, kernel=(1, 1), pad=(0, 0), name=('%s_conv1' % name))
    out = sym.BN_ACT_Conv(data=out,  num_filter=num_mid, kernel=(3, 3), pad=(1, 1), name=('%s_conv2' % name), stride=stride, num_group=num_group)
    out = sym.BN_ACT_Conv(data=out,  num_filter=num_out, kernel=(1, 1), pad=(0, 0), name=('%s_conv3' % name))
    
    # optional
    out = SE_Block(sym=sym, data=out, num_out=num_out, name=('%s_se' % name)) if use_se else out
    
    if first_block:
        data = sym.BN_ACT_Conv(data=data, num_filter=num_out, kernel=(1, 1), pad=(0, 0), name=('%s_conv4' % name), stride=stride)
    
    out = sym.ElementWiseSum(*[data, out], name=('%s_sum' % name))
    return out
    
