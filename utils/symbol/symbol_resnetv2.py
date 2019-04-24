# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import mxnet as mx
from symbol_basic import Conv_BN_ACT, BN_ACT
from symbol_resblock import BottleNeckV2
        
         
resenet_spec = { 26: {2: 2, 3:  2, 4:  2, 5: 2},
                 50: {2: 3, 3:  4, 4:  6, 5: 3},
                101: {2: 3, 3:  4, 4: 23, 5: 3},
                152: {2: 3, 3:  8, 4: 36, 5: 3},
                200: {2: 3, 3: 24, 4: 36, 5: 3}}
                
def get_before_pool(depth, group=1, scaler=1., ratio=-1, use_fp16=False, use_se=False):
    data = mx.symbol.Variable(name="data")
    data = mx.sym.Cast(data=data, dtype=np.float16) if use_fp16 else data

    # define resne(x)t
    num_in, num_mid, num_out = (64, int(64 * scaler), 256)
    k_sec = resenet_spec[depth]
    
    # ---------
    
    # stage 1
    conv1_x = Conv_BN_ACT(data=data, num_filter=num_in, kernel=(7, 7), pad=(3, 3), stride=(2, 2), name='conv1')
    conv1_x = mx.symbol.Pooling(data=conv1_x, pool_type="max", kernel=(3, 3), pad=(1, 1), stride=(2, 2), name='pool1')

    # stage 2
    for i in range(1,k_sec[2]+1):
        conv2_x = BottleNeckV2(data=(conv1_x if i==1 else conv2_x),
                               num_in=(num_in if i==1 else num_out),
                               num_mid=num_mid,
                               num_out=num_out,
                               name="L2_B%02d"%i,
                               first_block=(i==1),
                               num_group=group,
                               stride=(1,1),
                               ratio=ratio,
                               use_se=use_se)

    # stage 3
    num_in  = num_out
    num_mid = int(2*num_mid)
    num_out = int(2*num_out)
    for i in range(1,k_sec[3]+1):
        conv3_x = BottleNeckV2(data=(conv2_x if i==1 else conv3_x),
                               num_in=(num_in if i==1 else num_out),
                               num_mid=num_mid,
                               num_out=num_out,
                               name="L3_B%02d"%i,
                               first_block=(i==1),
                               num_group=group,
                               stride=((2,2) if (i==1) else (1,1)),
                               ratio=ratio,
                               use_se=use_se)

    # stage 4
    num_in  = num_out
    num_mid = int(2*num_mid)
    num_out = int(2*num_out)
    for i in range(1,k_sec[4]+1):
        conv4_x = BottleNeckV2(data=(conv3_x if i==1 else conv4_x),
                               num_in=(num_in if i==1 else num_out),
                               num_mid=num_mid,
                               num_out=num_out,
                               name="L4_B%02d"%i,
                               first_block=(i==1),
                               num_group=group,
                               stride=((2,2) if (i==1) else (1,1)),
                               ratio=ratio,
                               use_se=use_se)

    # stage 5
    num_in  = num_out
    num_mid = int(2*num_mid)
    num_out = int(2*num_out)
    for i in range(1,k_sec[5]+1):
        # ratio is forced to be 0. for the last stage
        # (because do 3x3 conv on 3.5x3.5 resolution map does not make sense)
        conv5_x = BottleNeckV2(data=(conv4_x if i==1 else conv5_x),
                               num_in=(num_in if i==1 else num_out),
                               num_mid=num_mid,
                               num_out=num_out,
                               name="L5_B%02d"%i,
                               first_block=(i==1),
                               num_group=group,
                               stride=((2,2) if (i==1) else (1,1)),
                               ratio=min(ratio,0.),
                               use_se=use_se)

    # ---------
    # output
    output = BN_ACT(conv5_x, name='tail') # remove this line for BottleNeckV1
    output = mx.sym.Cast(data=output, dtype=np.float32) if use_fp16 else output
    return output


def get_linear(num_classes, depth, group=1, scaler=1., ratio=-1, dropout=0., use_fp16=False, use_se=False):
    before_pool = get_before_pool(depth, group, scaler, ratio, use_fp16, use_se)
    # - - - - -
    pool5 = mx.symbol.Pooling(data=before_pool, pool_type="avg", kernel=(7, 7), stride=(1, 1), global_pool=True, name="global-pool")
    flat5 = mx.symbol.Flatten(data=pool5, name='flatten')
    # suggested p value: 0.5 for finetuning on small dataset; 0.1 for distributed training; 0 for single node training
    flat5 = mx.symbol.Dropout(data=flat5, p=dropout) if dropout > 0 else flat5
    fc6   = mx.symbol.FullyConnected(data=flat5, num_hidden=num_classes, name='classifier')
    return fc6

def get_symbol(num_classes, depth, group=1, scaler=1., ratio=-1, dropout=0., use_fp16=False, use_se=False):
    fc6     = get_linear(num_classes, depth, group, scaler, ratio, dropout, use_fp16, use_se)
    softmax = mx.symbol.SoftmaxOutput( data=fc6,  name='softmax')
    return softmax


# code for debugging and plot network architecture
if __name__ == '__main__':

    # settings
    depth = [26, 50, 101, 152, 200][0]
    ratio = [-1, 0.125, 0.25, 0.5, 0.75][0] # set -1 for baseline network
    data_shape = (1, 3, 224, 224)
    
    # settings
    sym = get_linear(num_classes=1000, depth=depth, ratio=ratio)
    sym.save('symbol-debug.json')
    
    # print on terminal
    mx.visualization.print_summary(sym, shape={'data': data_shape})
    
    # plot network architecture
    digraph = mx.viz.plot_network(sym,  shape={'data': data_shape}, save_format='png')
    digraph.render(filename='debug')
