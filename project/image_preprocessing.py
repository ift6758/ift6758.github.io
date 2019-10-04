import os
import tensorflow as tf
import tensorflow_hub as hub
from typing import *



@tf.function
def read_image_file(file_path: Union[str, tf.Tensor]) -> tf.Tensor:
    """Reads an image file and returns the tf.Tensor
    
    Arguments:
        file_path {Union[str, tf.Tensor]} -- the path, either a string or a string Tensor
    
    Returns:
        tf.Tensor -- the image, as a tf.float32 tensor.
    """
    file_path = tf.convert_to_tensor(file_path, dtype=tf.string)
    # load the raw data from the file as a string
    image = tf.io.read_file(file_path)
    # convert the compressed string to a 3D uint8 tensor
    image = tf.image.decode_jpeg(image, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

@tf.function
def get_image_for_user(input_dir: str, userid: str) -> tf.Tensor:
    """Returns the image associated with the given userid.
    
    Arguments:
        userid {tf.Tensor} -- the userid string (should also work with numpy or pandas element)
        
    Returns:
        tf.Tensor -- the image tensor.
    """
    image_path = tf.strings.join([input_dir, "/Image/", userid, ".jpg"])
    return read_image_file(image_path)

def get_image_preprocessing_fn() -> Callable:
    """Returns the function that will be used to preprocessing the images.

    NOTE: we do it this way in order to only load the hub module once, rather than on every preprocessing step.
    
    Returns:
        [type] -- [description]
    """
    image_embedding_module_url = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"
    embed_image = hub.KerasLayer(image_embedding_module_url, output_shape=[2048], trainable=False, name="image_features")
    
    # Image dimensions required by this particular hub module
    IMG_WIDTH = 299
    IMG_HEIGHT = 299

    @tf.function
    def preprocess_images(images):
        # resize the images to the desired size.
        resized_images = tf.image.resize(images, [IMG_WIDTH, IMG_HEIGHT])
        feature_vectors = embed_image(resized_images)
        return feature_vectors
    
    return preprocess_images

def make_dataset(input_dir: str, userids: List[str]) -> tf.data.Dataset:
    """Creates the image dataset for the given userid's.
    
    Arguments:
        input_dir {str} -- the parent input directory
        userids {List[str]} -- the list of userids
    
    Returns:
        tf.data.Dataset -- the preprocessed image dataset, where each entry is the feature vector.
    """
    userid_dataset = tf.data.Dataset.from_tensor_slices(userids)
    
    image_preprocessing_fn = get_image_preprocessing_fn()

    @tf.function
    def load_image(userid: tf.Tensor) -> tf.Tensor:
        return get_image_for_user(input_dir, userid)

    preprocessing_batch_size = 100
    image_dataset = (
        userid_dataset
        .map(load_image)
        .batch(preprocessing_batch_size)
        .map(image_preprocessing_fn)
        .unbatch()
    )
    return image_dataset
