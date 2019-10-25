#!/usr/bin/python


import tensorflow as tf
import argparse
from model import get_model, HyperParameters
from collections import namedtuple
from typing import *
from dataclasses import dataclass
import pandas as pd

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

HyperParameters.add_arguments(parser)

args = parser.parse_args()

hparams = HyperParameters.from_args(args)

from preprocessing_pipeline import preprocess_train

@dataclass
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

def train_input_pipeline(data_dir: str, hparams: HyperParameters):
    train_data = TrainData(*preprocess_train(data_dir, hparams.num_like_pages))

    # TODO: save the information that will be used in the testing phase to a file or something.
    features = train_data.train_features
    labels = train_data.train_labels
    features.drop(["noface", "multiface"], axis=1, inplace=True)
    
    mins, maxes = train_data.features_min_max
    with open("train_features_max.csv", "w") as f:    
        maxes.to_csv(f, header=True)
    with open("train_features_min.csv", "w") as f:
        mins.to_csv(f, header=True)
    with open("train_features_likes.csv", "w") as f:
        likes = train_data.likes_kept
        f.write(",".join(likes))
    
    column_names = list(features.columns)
    print("number of columns:", len(column_names))
    print(column_names)
    

    assert "faceID" not in column_names
    assert "userId" not in column_names
    assert "userid" not in column_names
    expected_num_columns = hparams.num_text_features + hparams.num_image_features + hparams.num_like_pages
    # assert len(column_names) == expected_num_columns, column_names
    
    image_features_start_index = column_names.index("faceRectangle_width")
    likes_features_start_index = column_names.index("headPose_yaw")  + 1   

    all_features = features.values
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


# train_dir = "~/Train"
# from preprocessing import make_dataset
# features_dataset, labels_dataset = make_dataset(train_dir)


my_model = get_model(hparams)
print(my_model)

dataset = train_input_pipeline("./debug_data", hparams)

my_model.fit(
    dataset,
    epochs=10,
)