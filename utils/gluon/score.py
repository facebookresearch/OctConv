# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Code is based on:
https://github.com/dmlc/gluon-cv/blob/master/scripts/classification/imagenet/train_imagenet.py
"""

import argparse, time, logging, os, math

import numpy as np
import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluoncv.data import imagenet

# from gluoncv.model_zoo import get_model
from utils import get_model

# CLI
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--data-dir', type=str, default='~/.mxnet/datasets/imagenet',
                        help='training and validation pictures to use.')
    parser.add_argument('--rec-train', type=str, default='~/.mxnet/datasets/imagenet/rec/train.rec',
                        help='the training data')
    parser.add_argument('--rec-train-idx', type=str, default='~/.mxnet/datasets/imagenet/rec/train.idx',
                        help='the index of training data')
    parser.add_argument('--rec-val', type=str, default='~/.mxnet/datasets/imagenet/rec/val.rec',
                        help='the validation data')
    parser.add_argument('--rec-val-idx', type=str, default='~/.mxnet/datasets/imagenet/rec/val.idx',
                        help='the index of validation data')
    parser.add_argument('--use-rec', action='store_true',
                        help='use image record iter for data input. default is false.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='data type for training. default is float32')
    parser.add_argument('--num-gpus', type=int, default=0,
                        help='number of gpus to use.')
    parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--mode', type=str,
                        help='mode in which to train the model. options are symbolic, imperative, hybrid')
    parser.add_argument('--model', type=str, default='resnet101_v1d_hi',
                        help='type of model to use. see vision_model for options.')
    parser.add_argument('--ratio', type=float, default=0.,
                        help='percentage of the low frequency part')
    parser.add_argument('--input-size', type=int, default=224,
                        help='size of the input image size. default is 224')
    parser.add_argument('--crop-ratio', type=float, default=0.875,
                        help='Crop ratio during validation. default is 0.875')
    parser.add_argument('--use-se', action='store_true',
                        help='use SE layers or not in resnext. default is false.')
    parser.add_argument('--batch-norm', action='store_true',
                        help='enable batch normalization or not in vgg. default is false.')
    parser.add_argument('--resume-params', type=str, default='',
                        help='path of parameters to load from.')
    opt = parser.parse_args()
    return opt


def main():
    opt = parse_args()

    streamhandler = logging.StreamHandler()

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(streamhandler)

    logger.info(opt)

    batch_size = opt.batch_size
    classes = 1000
    num_eval_samples = 50000

    num_gpus = opt.num_gpus
    batch_size *= max(1, num_gpus)
    context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    num_workers = opt.num_workers

    model_name = opt.model

    kwargs = {'ctx': context, 'classes': classes, 'ratio': opt.ratio, 'use_se': opt.use_se}


    net = get_model(model_name, **kwargs)
    net.cast(opt.dtype)
    if opt.resume_params is not '':
        net.load_parameters(opt.resume_params, ctx = context)

    # Two functions for reading data from record file or raw images
    def get_data_rec(rec_val, rec_val_idx, batch_size, num_workers):
        input_size = opt.input_size
        crop_ratio = opt.crop_ratio if opt.crop_ratio > 0 else 0.875
        resize = int(math.ceil(input_size / crop_ratio))
        mean_rgb = [123.68, 116.779, 103.939]
        std_rgb = [58.393, 57.12, 57.375]

        def batch_fn(batch, ctx):
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
            return data, label

        val_data = mx.io.ImageRecordIter(
            path_imgrec         = rec_val,
            path_imgidx         = rec_val_idx,
            preprocess_threads  = num_workers,
            shuffle             = False,
            batch_size          = batch_size,
            resize              = resize,
            data_shape          = (3, input_size, input_size),
            mean_r              = mean_rgb[0],
            mean_g              = mean_rgb[1],
            mean_b              = mean_rgb[2],
            std_r               = std_rgb[0],
            std_g               = std_rgb[1],
            std_b               = std_rgb[2],
        )
        return val_data, batch_fn

    def get_data_loader(data_dir, batch_size, num_workers):
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        input_size = opt.input_size
        crop_ratio = opt.crop_ratio if opt.crop_ratio > 0 else 0.875
        resize = int(math.ceil(input_size / crop_ratio))

        def batch_fn(batch, ctx):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            return data, label

        transform_test = transforms.Compose([
            transforms.Resize(resize, keep_ratio=True),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize
        ])

        val_data = gluon.data.DataLoader(
            imagenet.classification.ImageNet(data_dir, train=False).transform_first(transform_test),
            batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return val_data, batch_fn

    if opt.use_rec:
        val_data, batch_fn = get_data_rec(opt.rec_val, opt.rec_val_idx,
                                          batch_size, num_workers)
    else:
        val_data, batch_fn = get_data_loader(opt.data_dir, batch_size, num_workers)

    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)

    def test(ctx, val_data):
        if isinstance(ctx, mx.Context):
            ctx = [ctx]
        if opt.use_rec:
            val_data.reset()
        acc_top1.reset()
        acc_top5.reset()

        tic = time.time()
        for i, batch in enumerate(val_data):
            data, label = batch_fn(batch, ctx)
            outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]
            acc_top1.update(label, outputs)
            acc_top5.update(label, outputs)
            if (i+1) % 50 == 0:
                _, top1 = acc_top1.get()
                _, top5 = acc_top5.get()
                logger.info('[Batch %d][Image %d] validation: top1=%f top5=%f'%(i+1, (i+1)*batch_size, top1, top5))

        _, top1 = acc_top1.get()
        _, top5 = acc_top5.get()
        return top1, top5, time.time() - tic

    if opt.mode == 'hybrid':
        net.hybridize(static_alloc=True, static_shape=True)

    top1, top5, eclipse = test(context, val_data)
    logger.info('speed: %d samples/sec\ttime cost: %fs'%(num_eval_samples/eclipse, eclipse))
    logger.info('validation: top1=%f top5=%f'%(top1, top5))


if __name__ == '__main__':
    main()
