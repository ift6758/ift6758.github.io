#!/bin/bash

EXPERIMENT_NAME="best_model_01"

mkdir logs

MAX_EPOCHS=500
# Total epochs: 0050, val_loss: 12.778, log_dir: checkpoints/SGD_with_regularization/2019-11-02_00:19:03,
# hparams: HyperParameters(batch_size=256, num_layers=3, dense_units=64, activation='tanh', optimizer='sgd', learning_rate=0.01, l1_reg=0.01, l2_reg=0.01, num_like_pages=5000, use_dropout=False, dropout_rate=0.1, use_batchnorm=False)

python ./ift6758.github.io/project/train.py \
        --experiment_name $EXPERIMENT_NAME \
        --epochs $MAX_EPOCHS \
        --batch_size 256 \
        --num_layers 3 \
        --dense_units 64 \
        --activation tanh \
        --learning_rate 0.01 \
        --num_like_pages 5000 \
        --use_dropout False \
        --use_batchnorm False \
        --optimizer sgd \
        --l1_reg 0.01 \
        --l2_reg 0.01 \
        --validation_data_fraction 0.0 \
        >> "logs/$EXPERIMENT_NAME.txt"
