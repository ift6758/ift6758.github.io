#!/usr/bin/python


import argparse
import json
import os
from collections import namedtuple
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import *

import numpy as np
import pandas as pd
import simple_parsing
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from simple_parsing import ArgumentParser

from model import HyperParameters, get_model
from preprocessing_pipeline import preprocess_train

today_str = (datetime.now().strftime("%Y-%m-%d_%H:%M"))
from utils import DEBUG

print("DEBUGGING: ", DEBUG)

@dataclass()
class TrainConfig():
    log_dir: str = f"checkpoints/{today_str}"
    """
    The directory where the model checkpoints, as well as logs and event files should be saved at.
    """

    validation_data_fraction: float = 0.2
    """
    The fraction of all data corresponding to the validation set. (20% by default)
    """

    epochs: int = 50
    """Number of passes through the dataset"""   

    # train_features_min_max: Tuple[pd.DataFrame, pd.DataFrame] = field(init=False)
    # train_features_image_means: List[float] = field(init=False)


@dataclass()
class TrainData():
    
    train_features: pd.DataFrame
    """vectorized features scaled between 0 and 1
    for each user id in the training set, concatenated for all modalities
    (order = text + image + relation), with userid as DataFrame index.
    """
    features_min_max: Tuple[pd.DataFrame, pd.DataFrame]
    """series of min and max values of
    text + image features from train dataset, to be used to scale test data.
    Note that the multihot relation features do not necessitate scaling.
    """
    image_means: List[float]
    """
    means from oxford dataset to replace missing entries in oxford test set
    """
    likes_kept: pd.Index
    """ordered likes_ids to serve as columns for test set relation features matrix
    """
    train_labels: pd.DataFrame
    """labels ordered by userid (alphabetically)
    for the training set, with userids as index.
    """

def train_input_pipeline(data_dir: str, hparams: HyperParameters, train_config: TrainConfig) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    train_data = TrainData(*preprocess_train(data_dir, hparams.num_like_pages))

    features = train_data.train_features
    labels = train_data.train_labels

    features.drop(["noface", "multiface"], axis=1, inplace=True)
    

    mins, maxes = train_data.features_min_max

    with open(os.path.join(train_config.log_dir, "train_features_max.csv"), "w") as f:    
        # maxes.to_csv(f, header=True)
        f.write(",".join(str(v) for v in maxes))
    with open(os.path.join(train_config.log_dir, "train_features_min.csv"), "w") as f:
        # mins.to_csv(f, header=True)
        f.write(",".join(str(v) for v in mins))
    with open(os.path.join(train_config.log_dir, "train_features_image_means.csv"), "w") as f:
        image_means = train_data.image_means
        f.write(",".join(str(v) for v in image_means))
    

    if DEBUG:
        print("Writing dummy likes kept")
        with open(os.path.join(train_config.log_dir, "train_features_likes.csv"), "w") as f:
            f.write(",".join(str(v) for v in range(hparams.num_like_pages)))
    else:
        with open(os.path.join(train_config.log_dir, "train_features_likes.csv"), "w") as f:
            likes = train_data.likes_kept
            f.write(",".join(likes))
    
    column_names = list(features.columns)
    # print("number of columns:", len(column_names))
    
    assert "faceID" not in column_names
    assert "userId" not in column_names
    assert "userid" not in column_names
    expected_num_columns = hparams.num_text_features + hparams.num_image_features + hparams.num_like_pages
    # assert len(column_names) == expected_num_columns, column_names
    
    all_features = features.values
    validation_data_percentage = 0.2
    cutoff = int(all_features.shape[0] * validation_data_percentage)

    valid_features, valid_labels = features.values[:cutoff], labels[:cutoff]
    train_features, train_labels = features.values[cutoff:], labels[cutoff:]

    image_features_start_index = column_names.index("faceRectangle_width")
    likes_features_start_index = column_names.index("headPose_yaw")  + 1   

    def make_dataset(features: np.ndarray, labels: np.ndarray):
        text_features   = features[..., :image_features_start_index]
        image_features  = features[..., image_features_start_index:likes_features_start_index]
        likes_features  = features[..., likes_features_start_index:]

        # print(text_features.shape)
        # print(image_features.shape)
        # print(likes_features.shape)
        features_dataset = tf.data.Dataset.from_tensor_slices(
            {
                "text_features": text_features.astype("float32"),
                "image_features": image_features.astype("float32"),
                "likes_features": likes_features.astype("bool"),
            }
        )
        labels_dataset = tf.data.Dataset.from_tensor_slices({
            "userid": labels.index,
            "gender": labels.gender.astype("bool"),
            "age_group": labels.age_group,
            "ope": labels['ope'].astype("float32"),
            "con": labels['con'].astype("float32"),
            "ext": labels['ext'].astype("float32"),
            "agr": labels['agr'].astype("float32"),
            "neu": labels['neu'].astype("float32"),
        })
        return (tf.data.Dataset.zip((features_dataset, labels_dataset))
            .cache()
            .shuffle(5 * hparams.batch_size)
            .batch(hparams.batch_size)
        )
    
    train_dataset = make_dataset(train_features, train_labels)
    valid_dataset = make_dataset(valid_features, valid_labels)
    return train_dataset, valid_dataset

def train(train_dir: str, hparams: HyperParameters, train_config: TrainConfig):
    # Create the required directories if not present.
    os.makedirs(train_config.log_dir, exist_ok=True)

    # save the hyperparameter config to a file.
    with open(os.path.join(train_config.log_dir, "hyperparameters.json"), "w") as f:
        json.dump(asdict(hparams), f, indent=4)
    
    with open(os.path.join(train_config.log_dir, "train_config.json"), "w") as f:
        json.dump(asdict(train_config), f, indent=4)

    model = get_model(hparams)
    model.summary()

    train_dataset, valid_dataset = train_input_pipeline(train_dir, hparams, train_config)
    training_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(train_config.log_dir, "model.h5"),
            monitor = "val_loss",
            verbose=1,
            save_best_only=True,
            mode = 'auto'
        ),
        tf.keras.callbacks.TensorBoard(log_dir = train_config.log_dir, profile_batch=0),
        hp.KerasCallback(train_config.log_dir, asdict(hparams)),
    ]
    model.fit(
        train_dataset.repeat(100) if DEBUG else train_dataset,
        validation_data=valid_dataset,
        epochs=train_config.epochs,
        callbacks=training_callbacks
    )
    return model



if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_arguments(HyperParameters, "hparams")
    parser.add_arguments(TrainConfig, "train_config")

    args = parser.parse_args()
    
    hparams: HyperParameters = args.hparams
    train_config: TrainConfig = args.train_config
    
    
    print("Hyperparameters:", hparams)
    print("Train_config:", train_config)

    train_dir = "./debug_data" if DEBUG else "~/Train"
    model = train(train_dir, hparams, train_config)
    print(f"Saved model weights are located at '{train_config.log_dir}'")
    # save_path = os.path.join(train_config.log_dir, "model_final.h5")
    # model.save(save_path)
