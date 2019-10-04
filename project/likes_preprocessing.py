import tensorflow as tf
import tensorflow_hub as hub
from typing import *

def make_dataset(input_dir: str, userids: List[str]) -> tf.data.Dataset:
    """Creates the preprocessed text dataset for the given userid's.
    
    Arguments:
        input_dir {str} -- the parent input directory
        userids {List[str]} -- the list of userids
    
    Returns:
        tf.data.Dataset -- the preprocessed text dataset, where each entry is the feature vector.
    """
    # TODO
    raise NotImplementedError()

