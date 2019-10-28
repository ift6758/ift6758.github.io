import tensorflow as tf

from socket import gethostname
DEBUG = "fabrice" in gethostname()


def random_multihot_vector(num_examples, num_classes, prob_1: float = 0.5) -> tf.Tensor:
    """Creates a multi-hot random 'likes' vector.
    
    Keyword Arguments:
        prob_1 {float} -- the probability of having a '1' at each entry. (default: {0.5})
    
    Returns:
        tf.Tensor -- a multi-hot vector of shape [num_examples, num_like_pages], and of dtype tf.bool
    """
    return tf.cast(tf.random.categorical(
        logits=tf.math.log([[1 - prob_1, prob_1] for _ in range(num_examples)]),
        num_samples=num_classes,
        dtype=tf.int32,
    ), tf.bool)
