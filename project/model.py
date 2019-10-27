"""
@Fabrice: Here's an example of the kind of Model we could potentially use. (given what we know about the inputs we're gonna receive.)
"""

import tensorflow as tf
import dataclasses
from dataclasses import dataclass
from tensorboard.plugins.hparams import api as hp
# install simple-parsing with `pip install simple-parsing` 

@dataclass
class HyperParameters():
    # the batch size
    batch_size: int = 32
    # the number of dense layers in our model.
    num_layers: int = 10
    # the number of units in each dense layer.
    dense_units: int = 256
    
    # the activation function used after each dense layer
    activation: str = "tanh"

    # number of individual 'pages' that were kept during preprocessing of the 'likes'.
    # This corresponds to the number of entries in the multi-hot like vector.
    num_like_pages: int = 5
    # wether or not Dropout layers should be used
    use_dropout: bool = True
    # the dropout rate
    dropout_rate: float = 0.5

    num_text_features: int = 91
    num_image_features: int = 63




def get_model(hparams: HyperParameters) -> tf.keras.Model:
    # INPUTS: genderate content: (e.g., text, image and relations)
    # Inputs: user id (str), text + image info (157 floats), 
    # OUTPUTS:  Gender (ACC), Age (ACC), EXT (RMSE), ope (RMSE), AGR (RMSE), NEU (RMSE), CON (RMSE)
    # Texte: 82 (LIWC) + 11 (NRC)
    # Image
    
    # defining the inputs:
    # userid         =    tf.keras.Input([], dtype=tf.string, name="userid")
    image_features =    tf.keras.Input([hparams.num_image_features], dtype=tf.float32, name="image_features")
    text_features  =    tf.keras.Input([hparams.num_text_features], dtype=tf.float32, name="text_features")
    likes_features =    tf.keras.Input([hparams.num_like_pages], dtype=tf.bool, name="likes_features")

    # TODO: see below.
    likes_float = tf.cast(likes_features, tf.float32)

    # TODO: maybe use some kind of binary neural network here to condense a [`num_like_pages`] bool vector down to something more manageable (ex: [128] floats)
    likes_condensing_block = tf.keras.Sequential(name="likes_condensing_block")
    likes_condensing_block.add(tf.keras.layers.Dense(units=512, activation=hparams.activation))
    likes_condensing_block.add(tf.keras.layers.Dense(units=256, activation=hparams.activation))
    likes_condensing_block.add(tf.keras.layers.Dense(units=128, activation=hparams.activation))

    condensed_likes = likes_condensing_block(likes_float)

    # Dense block (applied on all the features, concatenated.)
    dense_layers = tf.keras.Sequential(name="dense_layers")
    dense_layers.add(tf.keras.layers.Concatenate())
    for i in range(hparams.num_layers):
        dense_layers.add(tf.keras.layers.Dense(units=hparams.dense_units, activation=hparams.activation))
        if hparams.use_dropout:
            dense_layers.add(tf.keras.layers.Dropout(hparams.dropout_rate))
    # get the dense feature representation
    features = dense_layers([text_features, image_features, condensed_likes])
    
    # MODEL OUTPUTS:
    age_group = tf.keras.layers.Dense(units=4, activation="softmax", name="age_group")(features)
    gender = tf.keras.layers.Dense(units=1, activation="sigmoid", name="gender")(features)
    
    def personality_scaling(name: str) -> tf.keras.layers.Layer:
        """Returns a layer that scales a sigmoid output [0, 1) output to the desired 'personality' range of [1, 5)
        
        Arguments:
            name {str} -- the name to give to the layer.
        
        Returns:
            tf.keras.layers.Layer -- the layer to use.
        """
        return tf.keras.layers.Lambda(lambda x: x * 4.0 + 1.0, name=name)

    ext_sigmoid = tf.keras.layers.Dense(units=1, activation="sigmoid", name="ext_sigmoid")(features)
    ext = personality_scaling("ext")(ext_sigmoid)

    ope_sigmoid = tf.keras.layers.Dense(units=1, activation="sigmoid", name="ope_sigmoid")(features)
    ope = personality_scaling("ope")(ope_sigmoid)
    
    agr_sigmoid = tf.keras.layers.Dense(units=1, activation="sigmoid", name="agr_sigmoid")(features)
    agr = personality_scaling("agr")(agr_sigmoid)
    
    neu_sigmoid = tf.keras.layers.Dense(units=1, activation="sigmoid", name="neu_sigmoid")(features)
    neu = personality_scaling("neu")(neu_sigmoid)
    
    con_sigmoid = tf.keras.layers.Dense(units=1, activation="sigmoid", name="con_sigmoid")(features)
    con = personality_scaling("con")(con_sigmoid)

    model = tf.keras.Model(
        inputs=[text_features, image_features, likes_features],
        outputs=[age_group, gender, ext, ope, agr, neu, con]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss={
            "age_group": tf.keras.losses.CategoricalCrossentropy(),
            "gender": "binary_crossentropy",
            "ext": "mse",
            "ope": "mse",
            "agr": "mse",
            "neu": "mse",
            "con": "mse",
        },
        #TODO: We can use this to change the importance of each output in the loss calculation, if need be.
        loss_weights={ 
            "age_group": 1,
            "gender": 1,
            "ext": 1,
            "ope": 1,
            "agr": 1,
            "neu": 1,
            "con": 1,
        },
        metrics={
            "age_group": tf.keras.metrics.CategoricalAccuracy(),
            "gender": tf.keras.metrics.BinaryAccuracy(),
            "ext": tf.keras.metrics.RootMeanSquaredError(),
            "ope": tf.keras.metrics.RootMeanSquaredError(),
            "agr": tf.keras.metrics.RootMeanSquaredError(),
            "neu": tf.keras.metrics.RootMeanSquaredError(),
            "con": tf.keras.metrics.RootMeanSquaredError(),
        },
    )
    # model.summary()
    return model