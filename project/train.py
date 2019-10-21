#!/usr/bin/python


import tensorflow as tf
import argparse
from model import get_model, HyperParameters
from typing import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

HyperParameters.add_arguments(parser)

args = parser.parse_args()

hparams = HyperParameters.from_args(args)

from preprocessing import random_dataset
features_dataset, labels_dataset = random_dataset()

# train_dir = "~/Train"
# from preprocessing import make_dataset
# features_dataset, labels_dataset = make_dataset(train_dir)

train_dataset = (tf.data.Dataset.zip((features_dataset, labels_dataset))
    .cache()
    .shuffle(5 * hparams.batch_size)
    .batch(hparams.batch_size)
)

my_model = get_model(hparams)
print(my_model)


my_model.fit(
    train_dataset,
    epochs=10,
)