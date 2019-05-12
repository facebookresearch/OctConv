# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
script for testing the pre-trained model
- ref: https://github.com/cypw/DPNs/blob/master/score.py
"""
import argparse
import mxnet as mx
import time
import os
import logging

def score(model, dataset, metrics, gpus, batch_size, rgb_mean, network,
          data_shape, epoch, scale=0.0167):
    # create data iterator
    rgb_mean = [float(i) for i in rgb_mean.split(',')]
    data_shape = tuple([int(i) for i in data_shape.split(',')])
    data = mx.io.ImageRecordIter(
        data_name          = 'data',
        label_name         = 'softmax_label',
        # ------------------------------------
        path_imgrec        = dataset,
        label_width        = 1,
        data_shape         = data_shape,
        preprocess_threads = 16,
        # ------------------------------------ 
        batch_size         = batch_size,
        # ------------------------------------ 
        mean_r             = rgb_mean[0],
        mean_g             = rgb_mean[1],
        mean_b             = rgb_mean[2],
        scale              = scale,
        # ------------------------------------
        rand_crop          = False,
        resize             = 256,
        inter_method       = 2 # bicubic
        )

    # load parameters
    sym, arg_params, aux_params = mx.model.load_checkpoint(model, epoch)
    logging.info('loading {}-{:04d}.params'.format(model, epoch))
    logging.info('loading {}-symbol.json'.format(model, epoch))

    # bind
    devs = mx.cpu() if not gpus else [mx.gpu(int(i)) for i in gpus.split(',')]
    mod = mx.mod.Module(symbol=sym, context=devs)
    mod.bind(for_training=False,
             data_shapes=data.provide_data,
             label_shapes=data.provide_label)
    mod.set_params(arg_params, aux_params)
    if not isinstance(metrics, list):
        metrics = [metrics,]

    # testing    
    num = 0
    for batch in data:
        tic = time.time()
        mod.forward(batch, is_train=False)
        for m in metrics[::-1]:
            mod.update_metric(m, batch.label)
        num += batch_size
        if num%10000==0:
            cost = 1000 * (time.time() - tic) / batch_size # ms
            logging.info('{}: {}, {:.1f} ms/im'.format(num,m.get(),cost))

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='score a model on a dataset')
    parser.add_argument('--model',       type=str, required=True,)
    parser.add_argument('--gpus',        type=str, default='1')
    parser.add_argument('--batch-size',  type=int, default=200)
    parser.add_argument('--epoch',       type=int, default=0)
    parser.add_argument('--rgb-mean',    type=str, default='124,117,104')
    parser.add_argument('--dataset',     type=str, default='/tmp/val.rec')
    parser.add_argument('--data-shape',  type=str, default='3,224,224')
    parser.add_argument('--network',     type=str)
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    metrics = [mx.metric.create('acc'),
               mx.metric.create('top_k_accuracy', top_k = 5)]

    score(metrics = metrics, **vars(args))
    logging.info('Finished')

    for m in metrics:
        logging.info(m.get())
