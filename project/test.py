#!/usr/bin/python


import argparse
import os
from collections import namedtuple
from dataclasses import dataclass
from typing import *

import pandas as pd
import tensorflow as tf

from model import HyperParameters, get_model
from preprocessing_pipeline import preprocess_test
from train import TrainConfig


def test_input_pipeline(data_dir: str, hparams: HyperParameters, train_config: TrainConfig):
    maxes = pd.read_csv(os.path.join(train_config.train_features_characteristics_save_dir, "train_features_max.csv"))
    mins = pd.read_csv(os.path.join(train_config.train_features_characteristics_save_dir, "train_features_min.csv"))
    with open(os.path.join(train_config.train_features_characteristics_save_dir,"train_features_likes.csv")) as f:
        likes_kept_train = ",".split(f.readline())
    with open(os.path.join(train_config.train_features_characteristics_save_dir,"train_features_image_means.csv")) as f:
        image_means_train = [float(v) for v in ",".split(f.readline())]
    min_max_train = (mins, maxes)

    test_features = preprocess_test(data_dir, min_max_train, image_means_train, likes_kept_train)

    # TODO: save the information that will be used in the testing phase to a file or something.
    column_names = list(test_features.columns)
    print("number of columns:", len(column_names))
    
    assert "faceID" not in column_names
    assert "userId" not in column_names
    assert "userid" not in column_names
    expected_num_columns = hparams.num_text_features + hparams.num_image_features + hparams.num_like_pages
    assert len(column_names) == expected_num_columns, column_names
    
    image_features_start_index = column_names.index("faceRectangle_width")
    likes_features_start_index = column_names.index("headPose_yaw")  + 1   

    all_features = test_features.values
    text_features = all_features[..., :image_features_start_index]
    image_features = all_features[..., image_features_start_index:likes_features_start_index]
    likes_features = all_features[..., likes_features_start_index:]

    # print(text_features.shape)
    # print(image_features.shape)
    # print(likes_features.shape)
    
    features_dataset = tf.data.Dataset.from_tensor_slices(
        {
            "text_features": text_features,
            "image_features": image_features,
            "likes_features": likes_features,
        }
    )
    return (features_dataset
        .cache()
        .batch(hparams.batch_size)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_model_dir", type=str, help="directory of the trained model to use for inference.")
    parser.add_argument("-i", type=str, default="./debug_data", help="Input directory")
    parser.add_argument("-o", type=str, default="./debug_output", help="Output directory")
    args = parser.parse_args()


    input_dir:  str = args.i
    output_dir: str = args.o
    
    trained_model_dir: str = args.trained_model_dir
    trained_model_weights_path = os.path.join(trained_model_dir, "model_final.h5")
    trained_model_hparams_path = os.path.join(trained_model_dir, "hyperparameters.json")
    trained_model_config_path = os.path.join(trained_model_dir, "train_config.json")
    import json
    import os

    with open(trained_model_hparams_path) as f:
        hparams = HyperParameters(**json.load(f))

    with open(trained_model_config_path) as f:
        train_config = TrainConfig(**json.load(f))

    model = get_model(hparams)
    model.load_weights(trained_model_weights_path)

    test_dataset = test_input_pipeline(input_dir, hparams, train_config)

    predictions = model.predict(test_dataset)
    for i, prediction in enumerate(predictions):
        print("prediction:", prediction)
