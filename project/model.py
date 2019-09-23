"""
@Fabrice: Here's an example of the kind of Model we could potentially use. (given what we know about the inputs we're gonna receive.)
"""

import tensorflow as tf
import tensorflow_hub as hub
print("tensorflow version:", tf.__version__)

# INPUTS: genderate content: (e.g., text, image and relations)
# OUTPUTS:  Gender (ACC), Age (ACC), EXT (RMSE), OPN (RMSE), AGR (RMSE), NEU (RMSE), CON (RMSE)

# defining the inputs:
text_in = tf.keras.Input((), dtype=tf.string, name="text_in")
image_in = tf.keras.Input((299,299,3), dtype=tf.float32, name="image_in")
""" NOTE: The 'relations' input field doesn't have a clear definition yet.

if "relations" is indeed a list of people which are somehow related to this individual,
then we should probably use a [Graph Convolutional Network (GCN)](https://arxiv.org/abs/1609.02907).

Then, "relations" input would be an Adjacency matrix between people.
"""
adjacency_matrix_size = 100
relations_in = tf.keras.Input(name="relations_in", shape=[adjacency_matrix_size, adjacency_matrix_size], dtype=bool)


# embedding models: These are pre-trained feature extractors available from tensorflow-hub:
text_embedding_module_url = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"
embed_text = hub.KerasLayer(text_embedding_module_url, output_shape=[128], input_shape=[], dtype=tf.string, trainable=False, name="text_features")
text_features = embed_text(text_in)

image_embedding_module_url = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"
embed_image = hub.KerasLayer(image_embedding_module_url, output_shape=[2048], trainable=False, name="image_features")
image_features = embed_image(image_in)


num_dense_layers = 2
dense_block = tf.keras.Sequential(name="dense_block")
dense_block.add(tf.keras.layers.Concatenate())
for i in range(num_dense_layers):
    dense_block.add(tf.keras.layers.Dense(units=256, activation="relu"))
    dense_block.add(tf.keras.layers.Dropout(0.5))


features = dense_block([text_features, image_features])


# max age is set to 125, for instance. 
# NOTE: maybe using a single number for the age would be better, since the loss could be 
max_age = 125

# MODEL OUTPUTS:
age_out = tf.keras.layers.Dense(units=max_age, activation="softmax", name="age_out")(features)
gender_out = tf.keras.layers.Dense(units=1, activation="sigmoid", name="gender_out")(features)
ext_out = tf.keras.layers.Dense(units=1, activation="sigmoid", name="ext_out")(features)
opn_out = tf.keras.layers.Dense(units=1, activation="sigmoid", name="opn_out")(features)
agr_out = tf.keras.layers.Dense(units=1, activation="sigmoid", name="agr_out")(features)
neu_out = tf.keras.layers.Dense(units=1, activation="sigmoid", name="neu_out")(features)
con_out = tf.keras.layers.Dense(units=1, activation="sigmoid", name="con_out")(features)

model = tf.keras.Model(
    inputs=[text_in, image_in, relations_in],
    outputs=[age_out, gender_out, ext_out, opn_out, agr_out, neu_out, con_out]
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss={
        "age_out": "sparse_categorical_crossentropy",
        "gender_out": "binary_crossentropy",
        "ext_out": "mse",
        "opn_out": "mse",
        "agr_out": "mse",
        "neu_out": "mse",
        "con_out": "mse",
    },
    #TODO: We can use this to change the importance of each output in the loss calculation, if need be.
    loss_weights={ 
        "age_out": 1,
        "gender_out": 2,
        "ext_out": 1,
        "opn_out": 1,
        "agr_out": 1,
        "neu_out": 1,
        "con_out": 1,
    },
    metrics={
        "age_out": "accuracy",
        "gender_out": "accuracy",
        "ext_out": tf.keras.metrics.RootMeanSquaredError(),
        "opn_out": tf.keras.metrics.RootMeanSquaredError(),
        "agr_out": tf.keras.metrics.RootMeanSquaredError(),
        "neu_out": tf.keras.metrics.RootMeanSquaredError(),
        "con_out": tf.keras.metrics.RootMeanSquaredError(),
    },
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            filepath='model.hdf5',
            monitor = "val_loss",
            verbose=1,
            save_best_only=True,
            mode = 'auto'
        ),
        tf.keras.callbacks.TensorBoard()
    ]
)
model.summary()