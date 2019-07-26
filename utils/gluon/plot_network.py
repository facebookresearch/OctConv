# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse, logging, json

import mxnet as mx
from utils import get_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--model', type=str, default='mobilenet_v1_075',
                        help='type of model to use. see vision_model for options.')
    parser.add_argument('--ratio', type=float, default=0.5,
                        help='percentage of the low frequency part')
    parser.add_argument('--use_se', action='store_true',
                        help='use SE layers or not in resnext. default is false.')
    opt = parser.parse_args()
    return opt


def main():
    opt = parse_args()

    kwargs = {'ctx': [mx.cpu()], 'pretrained': False, 'classes': 1000, 'ratio': opt.ratio}
    
    if opt.use_se:
        kwargs['use_se'] = True

    logging.info("get symbol ...")
    net = get_model(opt.model, **kwargs)

    # Option 1
    logging.info("option 1: print network ...")
    logging.info(net)

    # Option 2 (net must be HybridSequential, if want to plot whole graph)
    logging.info("option 2: draw network ...")
    net.hybridize()
    net.collect_params().initialize()

    x = mx.sym.var('data')
    sym = net(x)
    digraph = mx.viz.plot_network(sym, shape={'data':(1, 3, 224, 224)}, save_format = 'png')
    digraph.view()
    digraph.render()

    keys = sorted(dict(net.collect_params()).keys())
    logging.info(json.dumps(keys, indent=4))


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
