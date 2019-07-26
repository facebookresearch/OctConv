# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

. /private/home/yunpeng/anaconda3/etc/profile.d/conda.sh
conda activate mxnet-pip

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0

NUM_GPUS=2
BATCH_SIZE=100
REC_ROOT=/private/home/yunpeng/dataset/rec/org/imagenet_mxnet-official
MODEL_ROOT='./models'

eval_mobilenet_v1_075_alpha_0375 () {
    MODEL_NAME='mobilenet_v1_075'
    RATIO=0.375
    USE_SE=''
    MODEL_PATH=${MODEL_ROOT}/mobilenet_v1_075_alpha-0.375.params
}
eval_mobilenet_v1_100_alpha_0500 () {
    MODEL_NAME='mobilenet_v1_100'
    RATIO=0.5
    USE_SE=''
    MODEL_PATH=${MODEL_ROOT}/mobilenet_v1_100_alpha-0.5.params
}
eval_mobilenet_v2_100_alpha_0375 () {
    MODEL_NAME='mobilenet_v2_100'
    RATIO=0.375
    USE_SE=''
    MODEL_PATH=${MODEL_ROOT}/mobilenet_v2_100_alpha-0.375.params
}
eval_mobilenet_v2_1125_alpha_0500 () {
    MODEL_NAME='mobilenet_v2_1125'
    RATIO=0.5
    USE_SE=''
    MODEL_PATH=${MODEL_ROOT}/mobilenet_v2_1125_alpha-0.5.params
}
eval_resnet152_v1f_alpha_0125 () {
    MODEL_NAME='resnet152_v1f'
    RATIO=0.125
    USE_SE=''
    MODEL_PATH=${MODEL_ROOT}/resnet152_v1f_alpha-0.125.params
}
eval_seresnet152_v1e_alpha_0125 () {
    MODEL_NAME='resnet152_v1e'
    RATIO=0.125
    USE_SE='--use-se'
    MODEL_PATH=${MODEL_ROOT}/se-resnet152_v1e_alpha-0.125.params
}


# options:
eval_mobilenet_v1_075_alpha_0375
# eval_mobilenet_v1_100_alpha_0500
# eval_mobilenet_v2_100_alpha_0375
# eval_mobilenet_v2_1125_alpha_0500
# eval_resnet152_v1f_alpha_0125
# eval_seresnet152_v1e_alpha_0125

python score.py \
  --rec-val ${REC_ROOT}/val.rec --rec-val-idx ${REC_ROOT}/val.idx \
  --model ${MODEL_NAME} --mode hybrid --ratio ${RATIO} ${USE_SE} \
  --batch-size ${BATCH_SIZE} --num-gpus ${NUM_GPUS} -j 40 \
  --resume-params ${MODEL_PATH} \
  --use-rec --dtype float16 \
  2>&1 | tee -a x224_${MODEL_NAME}_alpha-${RATIO}.log
