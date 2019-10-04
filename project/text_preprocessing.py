import tensorflow as tf
import tensorflow_hub as hub
from typing import *


def get_text_dataset(input_dir: str, userids: List[str]) -> tf.data.Dataset:
    """Returns a tf.data.Dataset of the text for each user in `userids`.
    
    Arguments:
        input_dir {str} -- The parent input directory.
        userids {List[str]} -- the list of userids. the output dataset should have the same ordering as this list.
    
    Returns:
        tf.data.Dataset -- a tf.data.Dataset of strings #* Fab: Preferably a tf.data.Dataset. Might also work with pandas directly, I'm not sure.
    """
    raise NotImplementedError() # TODO



def get_text_preprocessing_fn() -> Callable:
    # embedding models: These are pre-trained feature extractors available from tensorflow-hub:
    text_embedding_module_url = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"
    embed_text = hub.KerasLayer(text_embedding_module_url, output_shape=[128], input_shape=[], dtype=tf.string, trainable=False, name="text_features")
    @tf.function
    def preprocess_text(text):
        return embed_text
    
    return preprocess_text


def make_dataset(input_dir: str, userids: List[str]) -> tf.data.Dataset:
    """Creates the preprocessed text dataset for the given userid's.
    
    Arguments:
        input_dir {str} -- the parent input directory
        userids {List[str]} -- the list of userids
    
    Returns:
        tf.data.Dataset -- the preprocessed text dataset, where each entry is the feature vector.
    """
    userid_dataset = tf.data.Dataset.from_tensor_slices(userids)
    
    text_preprocessing_fn = get_text_preprocessing_fn()
    preprocessing_batch_size = 100

    text_dataset = (
        get_text_dataset(input_dir, userids)
        .batch(preprocessing_batch_size)
        .map(text_preprocessing_fn)
        .unbatch()
    )
    return text_dataset
