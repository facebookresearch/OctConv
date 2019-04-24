# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import mxnet as mx

from symbol_basic import Conv_BN_ACT, BN_ACT
from symbol_denseblock import DenseBlock, Connector


densenet_spec = {121: (64, 32, {2: 6, 3: 12, 4: 24, 5: 16}),
                 161: (96, 48, {2: 6, 3: 12, 4: 36, 5: 24}),
                 169: (64, 32, {2: 6, 3: 12, 4: 32, 5: 32}),
                 201: (64, 32, {2: 6, 3: 12, 4: 48, 5: 32})}


def get_before_pool(depth, ratio=-1, use_fp16=False):
    data = mx.symbol.Variable(name="data")
    data = mx.sym.Cast(data=data, dtype=np.float16) if use_fp16 else data
    
    # define densenet
    width, inc, k_sec = densenet_spec[depth]
    mid = int(4*inc)

    # ---------
    
    # stage 1
    conv1_x = Conv_BN_ACT(data=data, num_filter=width, kernel=(7, 7), pad=(3, 3), stride=(2, 2), name='conv1')
    conv2_x = mx.symbol.Pooling(data=conv1_x, pool_type="max", kernel=(3, 3), pad=(1,1), stride=(2, 2), name='pool1')
    
    conv2_x = (conv2_x, None) if ratio >= 0 else conv2_x

    # stage 2
    for i in range(1, k_sec[2]+1):
        conv2_x = DenseBlock(conv2_x, mid, inc, 'L2_B%02d'%i, ratio=ratio)
        width += inc

                
    # stage 3
    width = int(width/2)
    conv3_x = Connector(conv2_x, width, 'L3_Trans', ratio=ratio)
    for i in range(1, k_sec[3]+1):
        conv3_x = DenseBlock(conv3_x, mid, inc, 'L3_B%02d'%i, ratio=ratio)
        width += inc

               
    # stage 4
    width = int(width/2)
    conv4_x = Connector(conv3_x, width, 'L4_Trans', ratio=ratio)
    for i in range(1, k_sec[4]+1):
        conv4_x = DenseBlock(conv4_x, mid, inc, 'L4_B%02d'%i, ratio=ratio)
        width += inc

             
    # stage 5
    width = int(width/2)
    conv5_x = Connector(conv4_x, width, 'L5_Trans', ratio=min(ratio,0.)) # force to 0. because the shape is 7x7
    for i in range(1, k_sec[5]+1):
        # ratio is forced to be 0. for the last stage
        # (because do 3x3 conv on 3.5x3.5 resolution map does not make sense)
        conv5_x = DenseBlock(conv5_x, mid, inc, 'L5_B%02d'%i, ratio=min(ratio,0.))
        width += inc


    # ---------
    # output
    output = BN_ACT(data=conv5_x, name="tail")
    output = mx.sym.Cast(data=output, dtype=np.float32) if use_fp16 else output
    return output


def get_linear(num_classes, depth, ratio=-1, dropout=0., use_fp16=False):
    before_pool = get_before_pool(depth, ratio, use_fp16)
    # - - - - -
    pool5 = mx.symbol.Pooling(data=before_pool, pool_type="avg", kernel=(7, 7), stride=(1, 1), global_pool=True, name="global-pool")
    flat5 = mx.symbol.Flatten(data=pool5, name='flatten')
    # suggested p value: 0.5 for finetuning on small dataset; 0.1 for distributed training; 0 for single node training
    flat5 = mx.symbol.Dropout(data=flat5, p=dropout) if dropout > 0 else flat5
    fc6   = mx.symbol.FullyConnected(data=flat5, num_hidden=num_classes, name='classifier')
    return fc6

def get_symbol(num_classes, depth, ratio=-1, dropout=0., use_fp16=False):
    fc6     = get_linear(num_classes, depth, ratio, dropout, use_fp16)
    softmax = mx.symbol.SoftmaxOutput(data=fc6, name='softmax')
    return softmax


# code for debugging and plot network architecture
if __name__ == '__main__':

    # settings
    depth = [121, 161, 169, 201][0]
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
