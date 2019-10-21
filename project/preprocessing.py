"""
Voici ce que j'ai présentement comme description de notre dataset. @marie s'il te plait return ton dataset dans ce format là? (ou du moins quelque chose de semblable?)
# ```
# TODO = -1
# features = tf.data.Dataset.zip({
#     "userids": userids_dataset, # [9500, 1] vector of strings
#     "image_features": image_features_dataset,  #[9500, ?] vector of floats, same ordering as userids
#     "text_features": text_features_dataset, # [9500, ?]  vector of floats, same ordering as userids
#     "likes": preprocessed_likes, # [9500, 10 000], vector of bools/ints/whatever, same ordering as userids
# })
# labels = tf.data.Dataset.zip({
#     "age_group": TODO, #[9500, 4] one-hot vector for the four age groups
#     "gender": TODO, # [9500, 2] one-hot vector for gender
#     "ope": TODO, # [9500 , 1] vector of floats in range [1, 5)
#     "con": TODO, # [9500 , 1] vector of floats in range [1, 5)
#     "ext": TODO, # [9500 , 1] vector of floats in range [1, 5)
#     "agr": TODO, # [9500 , 1] vector of floats in range [1, 5)
#     "neu": TODO, # [9500 , 1] vector of floats in range [1, 5)
# })
# ```

Returns:
    [type] -- [description]
"""
import os
import glob
import tensorflow as tf
import tensorflow_hub as hub
from typing import *

import image_preprocessing
import text_preprocessing
import likes_preprocessing

def get_user_ids(input_dir: str) -> List[str]:
    """Given an input_dir, returns the list of userids. This is to be used as the reference, in terms of ordering, for all other datasets.

    Arguments:
        input_dir {str} -- The parent input directory (ex, "~/Train"), should have a "Image" subfolder with image files.

    Returns:
        List[str] -- [description]
    """
    image_files_name_pattern = os.path.join(input_dir, "Image", "*.jpg")
    image_filepaths = glob.glob(image_files_name_pattern)
    file_names = (os.path.basename(file_path) for file_path in image_filepaths)
    userids = [file_name.split(".")[0] for file_name in file_names]
    userids = sorted(userids)
    return userids


def make_dataset(input_dir: str) -> tf.data.Dataset:
    """Creates the main dataset

    Arguments:
        input_dir {str} -- The parent input directory (for example, "~/Train").

    Returns:
        tf.data.Dataset -- A `tf.data.Dataset` containing the userids and the processed versions of the images (feature vector), the text (feature vector) and the likes (one-hot binary vector). 
    """
    preprocessing_batch_size = 100
    userids = get_user_ids(input_dir)
    
    userids_dataset = tf.data.Dataset.from_tensor_slices(userids)
    preprocessed_images = image_preprocessing.make_dataset(input_dir, userids)
    # preprocessed_texts = text_preprocessing.make_dataset(input_dir, userids)
    # preprocessed_likes = likes_preprocessing.make_dataset(input_dir, userids)
    
    return tf.data.Dataset.zip({
        "userids": userids_dataset,
        "image_in": preprocessed_images,
        # "text_in": preprocessed_texts,
        # "likes_in": preprocessed_likes,
    })


def random_dataset(num_examples=9500, num_like_pages=10_000) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Creates a random dataset to use for testing/debugging.
    
    Keyword Arguments:
        num_examples {int} -- The number of users (default: {9500})
        num_like_pages {int} -- The number of liked 'pages' that are to be kept in preprocessing (default: {10_000})
    
    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset] -- a tuple of the (training_dataset, testing_dataset)
    """
    num_text_features = 91
    num_image_features = 65

    to_dataset = lambda x: tf.data.Dataset.from_tensor_slices(x)
    from utils import random_multihot_vector
    
    features = tf.data.Dataset.zip({
        "userids": to_dataset([str(i) for i in range(num_examples)]),
        "image_features": to_dataset(tf.random.uniform([num_examples, num_image_features])),
        "text_features": to_dataset(tf.random.uniform([num_examples, num_text_features])), 
        "likes": to_dataset(random_multihot_vector(num_examples, num_like_pages, prob_1=0.1)),
    })
    labels = tf.data.Dataset.zip({
        "age_group": to_dataset(tf.one_hot(
                indices = tf.random.uniform([num_examples], minval=0, maxval=4, dtype=tf.int32),
                depth=4,
            )),
        "gender": to_dataset(tf.random.uniform([num_examples, 1], maxval=2, dtype=tf.int32)),
        "ope": to_dataset(tf.random.uniform([num_examples, 1], minval=1.0, maxval=5.0)),
        "con": to_dataset(tf.random.uniform([num_examples, 1], minval=1.0, maxval=5.0)),
        "ext": to_dataset(tf.random.uniform([num_examples, 1], minval=1.0, maxval=5.0)),
        "agr": to_dataset(tf.random.uniform([num_examples, 1], minval=1.0, maxval=5.0)),
        "neu": to_dataset(tf.random.uniform([num_examples, 1], minval=1.0, maxval=5.0)),
    })
    return features, labels

if __name__ == "__main__":
    features, labels = random_dataset()
    for i, features_dict in enumerate(features):
        print(features_dict)
        break
