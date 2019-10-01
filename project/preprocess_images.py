import tensorflow as tf
import tensorflow_hub as hub

image_dir = "/home/fabrice/Pictures"
file_names = tf.data.Dataset.list_files(image_dir + "/*.png")

IMG_WIDTH = 299
IMG_HEIGHT = 299
BATCH_SIZE = 32

def get_user_id(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, '/')
    filename = tf.strings.split(parts[-1], ".")[0]
    return filename

def process_path(file_path):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_image(img)
    return img

def decode_image(image):
    # convert the compressed string to a 3D uint8 tensor
    image = tf.image.decode_jpeg(image, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    image = tf.image.convert_image_dtype(image, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(image, [IMG_WIDTH, IMG_HEIGHT])

image_embedding_module_url = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"
embed_image = hub.KerasLayer(image_embedding_module_url, output_shape=[2048], trainable=False, name="image_features")

dataset = (
        file_names
        # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
        .map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(BATCH_SIZE)
        .map(embed_image)
        .unbatch()
)
# writer = tf.data.experimental.TFRecordWriter("./images.tfrecord")
# writer.write(dataset.map(lambda ex: tf.train.FloatList(ex).))

# images = tf.data.TFRecordDataset("./images.tfrecord")

# TODO: Write out the numpy arrays to a file of some sort.
for i, image in enumerate(dataset):
    image = image.numpy()
