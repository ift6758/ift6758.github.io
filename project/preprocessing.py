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

if __name__ == "__main__":
    ds = make_dataset("./debug_data")
    for i, features in enumerate(ds):
        print(features)
    print(ds)
