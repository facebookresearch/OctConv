# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import symbol_resnetv2

group = 32
scaler = 2 # 32x4d increased the #channels by 2x (compared with 64x1d)
      
def get_before_pool(depth, ratio=-1, dropout=0., use_fp16=False):
    out = symbol_resnetv2.get_before_pool(depth=depth, 
                                          group=group,
                                          scaler=scaler,
                                          ratio=ratio,
                                          dropout=dropout,
                                          use_fp16=use_fp16)
    return out

def get_linear(num_classes, depth, ratio=-1, dropout=0., use_fp16=False):
    out = symbol_resnetv2.get_linear(num_classes=num_classes, 
                                     depth=depth,
                                     group=group,
                                     scaler=scaler,
                                     ratio=ratio,
                                     dropout=dropout,
                                     use_fp16=use_fp16)
    return out

def get_symbol(num_classes, depth, ratio=-1, dropout=0., use_fp16=False):
    out = symbol_resnetv2.get_symbol(num_classes=num_classes, 
                                     depth=depth,
                                     group=group,
                                     scaler=scaler,
                                     ratio=ratio,
                                     dropout=dropout,
                                     use_fp16=use_fp16)
    return out


# code for debugging and plot network architecture
if __name__ == '__main__':
    import mxnet as mx

    # settings
    depth = [26, 50, 101, 152, 200][0]
    ratio = [-1, 0.25, 0.5, 0.75][0] # set -1 for baseline network
    data_shape = (1, 3, 224, 224)
    
    # settings
    sym = get_linear(num_classes=1000, depth=depth, ratio=ratio)
    sym.save('symbol-debug.json')
    
    # print on terminal
    mx.visualization.print_summary(sym, shape={'data': data_shape})
    
    # plot network architecture
    digraph = mx.viz.plot_network(sym,  shape={'data': data_shape}, save_format='png')
    digraph.render(filename='debug')
