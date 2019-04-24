# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

__all__ = ['SE_Block',
          ]


'''see: Squeeze-and-Excitation Networks'''
def SE_Block(sym, data, num_out, name):

    if type(num_out) is tuple:
        num_mid = (int(sum(num_out)/16), 0)
    else:
        num_mid = int(num_out/16)

    # global pooling
    out = sym.Pooling(data=data, pool_type='avg', kernel=(1, 1), global_pool=True, stride=(1, 1), name=('%s_pool' % name))
    
    # fc1
    out = sym.FullyConnected(data=out, num_hidden=num_mid, no_bias=False, name=('%s_fc1' % name))
    out = sym.Activation(data=out, act_type='relu', name=('%s_relu' % name))
    
    # fc2
    out = sym.FullyConnected(data=out, num_hidden=num_out, no_bias=False, name=('%s_fc2' % name))
    out = sym.Activation(data=out, act_type='sigmoid', name=('%s_sigmoid' % name))
    
    # rescale
    out = sym.expand_dims(out, axis=2, name=('%s_expend1' % name))
    out = sym.expand_dims(out, axis=3, name=('%s_expend2' % name)) # add spatial dims back
    output = sym.broadcast_mul(data, out, name=('%s_mul' % name))
    
    return output

