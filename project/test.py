#!/usr/bin/python


import argparse
import os
from collections import namedtuple
from dataclasses import dataclass
from typing import *

import numpy as np
import pandas as pd
import tensorflow as tf

from model import HyperParameters, get_model
from preprocessing_pipeline import preprocess_test
from train import TrainConfig


def test_input_pipeline(data_dir: str, hparams: HyperParameters, train_config: TrainConfig):
    with open(os.path.join(train_config.log_dir, "train_features_max.csv")) as f:
        maxes = np.asarray([float(v)
                           for v in f.readline().split(",") if v.strip() != ""])
    with open(os.path.join(train_config.log_dir, "train_features_min.csv")) as f:
        mins = np.asarray([float(v)
                          for v in f.readline().split(",") if v.strip() != ""])
    with open(os.path.join(train_config.log_dir, "train_features_likes.csv")) as f:
        likes_kept_train = [int(like) for like in f.readline().split(",") if like.strip() != ""]
    with open(os.path.join(train_config.log_dir, "train_features_image_means.csv")) as f:
        parts = [float(v) for v in f.readline().split(",") if v.strip() != ""]
        image_means_train = parts
    min_max_train = (mins, maxes)

    test_features = preprocess_test(
        data_dir, min_max_train, image_means_train, likes_kept_train)
    test_features.drop(["noface", "multiface"], axis=1, inplace=True)

    # TODO: save the information that will be used in the testing phase to a file or something.
    column_names = list(test_features.columns)
    print("number of columns:", len(column_names))
    train_columns = [
        'WC', 'WPS', 'Sixltr', 'Dic', 'Numerals', 'funct', 'pronoun', 'ppron',
        'i', 'we', 'you', 'shehe', 'they', 'ipron', 'article', 'verb', 'auxverb',
        'past', 'present', 'future', 'adverb', 'preps', 'conj', 'negate', 'quant',
        'number', 'swear', 'social', 'family', 'friend', 'humans', 'affect', 'posemo',
        'negemo', 'anx', 'anger', 'sad', 'cogmech', 'insight', 'cause', 'discrep',
        'tentat', 'certain', 'inhib', 'incl', 'excl', 'percept', 'see', 'hear',
        'feel', 'bio', 'body', 'health', 'sexual', 'ingest', 'relativ', 'motion',
        'space', 'time', 'work', 'achieve', 'leisure', 'home', 'money', 'relig',
        'death', 'assent', 'nonfl', 'filler', 'Period', 'Comma', 'Colon', 'SemiC',
        'QMark', 'Exclam', 'Dash', 'Quote', 'Apostro', 'Parenth', 'OtherP', 'AllPct',
        'positive', 'negative', 'anger', 'anticipation', 'disgust', 'fear', 'joy',
        'sadness', 'surprise', 'trust', 'faceRectangle_width', 'faceRectangle_height',
        'faceRectangle_left', 'faceRectangle_top', 'pupilLeft_x', 'pupilLeft_y', 'pupilRight_x',
        'pupilRight_y', 'noseTip_x', 'noseTip_y', 'mouthLeft_x', 'mouthLeft_y', 'mouthRight_x',
        'mouthRight_y', 'eyebrowLeftOuter_x', 'eyebrowLeftOuter_y', 'eyebrowLeftInner_x',
        'eyebrowLeftInner_y', 'eyeLeftOuter_x', 'eyeLeftOuter_y', 'eyeLeftTop_x', 'eyeLeftTop_y',
        'eyeLeftBottom_x', 'eyeLeftBottom_y', 'eyeLeftInner_x', 'eyeLeftInner_y', 'eyebrowRightInner_x',
        'eyebrowRightInner_y', 'eyebrowRightOuter_x', 'eyebrowRightOuter_y', 'eyeRightInner_x',
        'eyeRightInner_y', 'eyeRightTop_x', 'eyeRightTop_y', 'eyeRightBottom_x', 'eyeRightBottom_y',
        'eyeRightOuter_x', 'eyeRightOuter_y', 'noseRootLeft_x', 'noseRootLeft_y', 'noseRootRight_x',
        'noseRootRight_y', 'noseLeftAlarTop_x', 'noseLeftAlarTop_y', 'noseRightAlarTop_x',
        'noseRightAlarTop_y', 'noseLeftAlarOutTip_x', 'noseLeftAlarOutTip_y', 'noseRightAlarOutTip_x',
        'noseRightAlarOutTip_y', 'upperLipTop_x', 'upperLipTop_y', 'upperLipBottom_x', 'upperLipBottom_y',
        'underLipTop_x', 'underLipTop_y', 'underLipBottom_x', 'underLipBottom_y', 'facialHair_mustache',
        'facialHair_beard', 'facialHair_sideburns', 'headPose_roll', 'headPose_yaw',
        # the like columns aren't here, obviously.
    ]


    assert "faceID" not in column_names
    assert "userId" not in column_names
    assert "userid" not in column_names
    expected_num_columns= hparams.num_text_features + hparams.num_image_features + hparams.num_like_pages

    # message = f"columnds present in train set but not in test set: {set(train_columns) ^ set(column_names)}"
    # assert len(column_names) == expected_num_columns, message
    image_features_start_index=column_names.index("faceRectangle_width")
    likes_features_start_index=column_names.index("headPose_yaw") + 1

    all_features=test_features.values
    text_features=all_features[..., : image_features_start_index]
    image_features=all_features[..., image_features_start_index: likes_features_start_index]
    likes_features = all_features[..., likes_features_start_index:]
    
    # print(text_features.shape, text_features.dtype)
    # print(image_features.shape, image_features.dtype)
    # print(likes_features.shape, likes_features.dtype)
    features_dataset= tf.data.Dataset.from_tensor_slices(
        {
            "userid": test_features.index.astype(str),
            "text_features": text_features.astype(float),
            "image_features": image_features.astype(float),
            "likes_features": likes_features.astype(bool),
        }
    )
    return features_dataset.batch(hparams.batch_size)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--trained_model_dir",
        type = str,
        default = "checkpoints/best_model_01/2019-11-02_01:05:33",
        help = "directory of the trained model to use for inference."
    )
    parser.add_argument(
        "-i", type = str, default = "./debug_data", help = "Input directory")
    parser.add_argument(
        "-o", type = str, default = "./debug_output", help = "Output directory")
    args=parser.parse_args()


    input_dir:  str=args.i
    output_dir: str=args.o

    trained_model_dir: str=args.trained_model_dir
    trained_model_weights_path=os.path.join(trained_model_dir, "model.h5")
    trained_model_hparams_path=os.path.join(
        trained_model_dir, "hyperparameters.json")
    trained_model_config_path=os.path.join(
        trained_model_dir, "train_config.json")
    
    import json
    import os

    with open(trained_model_hparams_path) as f:
        hparams=HyperParameters(**json.load(f))

    with open(trained_model_config_path) as f:
        train_config=TrainConfig(**json.load(f))

    model=get_model(hparams)
    model.load_weights(trained_model_weights_path)

    test_dataset=test_input_pipeline(input_dir, hparams, train_config)

    pred_labels = ["age_group", "gender", "ext", "ope", "agr", "neu", "con"]

    predictions=model.predict(test_dataset)
    print(len(predictions), "predictions")
    from user import User
    
    for i, user in enumerate(test_dataset.unbatch()):
        pred_dict = dict(zip(pred_labels, [p[i] for p in predictions]))
        userid = user["userid"].numpy().decode("utf-8")
        pred_dict["userid"] = userid
        pred_dict["age_group_id"] = np.argmax(pred_dict.pop("age_group"))
        pred_dict["gender"] = int(np.round(pred_dict["gender"]))
        
        user = User(**pred_dict)
        print(user)
        with open(os.path.join(output_dir, f"{userid}.xml"), "w") as xml_file:
            xml_file.write(user.to_xml())
