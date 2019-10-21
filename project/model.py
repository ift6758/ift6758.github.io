"""
@Fabrice: Here's an example of the kind of Model we could potentially use. (given what we know about the inputs we're gonna receive.)
"""

import tensorflow as tf
import dataclasses
from dataclasses import dataclass
# install simple-parsing with `pip install simple-parsing` 
from simple_parsing import ParseableFromCommandLine

@dataclass
class HyperParameters(ParseableFromCommandLine):
    # the batch size
    batch_size: int = 32
    # the number of dense layers in our model.
    num_layers: int = 10
    # the number of units in each dense layer.
    dense_units: int = 256
    
    # the activation function used after each dense layer
    activation: str = "relu"

    # number of individual 'pages' that were kept during preprocessing of the 'likes'.
    # This corresponds to the number of entries in the multi-hot like vector.
    num_like_pages: int = 10_000
    # wether or not Dropout layers should be used
    use_dropout: bool = True
    # the dropout rate
    dropout_rate: float = 0.5



def get_model(hparams: HyperParameters) -> tf.keras.Model:
    # INPUTS: genderate content: (e.g., text, image and relations)
    # Inputs: user id (str), text + image info (157 floats), 
    # OUTPUTS:  Gender (ACC), Age (ACC), EXT (RMSE), ope (RMSE), AGR (RMSE), NEU (RMSE), CON (RMSE)
    # Texte: 82 (LIWC) + 11 (NRC)
    # Image
    
    # defining the inputs:
    num_image_features = 65
    image_features = tf.keras.Input([num_image_features], dtype=tf.float32, name="image_features")
    
    num_text_features = 91
    text_features = tf.keras.Input([num_text_features], dtype=tf.float32, name="text_features")

    likes = tf.keras.Input(name="likes", shape=[hparams.num_like_pages], dtype=tf.bool)
    # TODO: maybe use some kind of binary neural network on the likes? casting booleans to floats is so ineficient!
    likes_float = tf.cast(likes, tf.float32)

    dense_layers = tf.keras.Sequential(name="dense_layers")
    dense_layers.add(tf.keras.layers.Concatenate())

    for i in range(hparams.num_layers):
        dense_layers.add(tf.keras.layers.Dense(units=hparams.dense_units, activation=hparams.activation))
        
        if hparams.use_dropout:
            dense_layers.add(tf.keras.layers.Dropout(hparams.dropout_rate))

    features = dense_layers([text_features, image_features, likes_float])
    
    
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
        inputs=[text_features, image_features, likes],
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
            "age_group": "accuracy",
            "gender": "accuracy",
            "ext": tf.keras.metrics.RootMeanSquaredError(),
            "ope": tf.keras.metrics.RootMeanSquaredError(),
            "agr": tf.keras.metrics.RootMeanSquaredError(),
            "neu": tf.keras.metrics.RootMeanSquaredError(),
            "con": tf.keras.metrics.RootMeanSquaredError(),
        },
        # callbacks=[
        #     tf.keras.callbacks.ModelCheckpoint(
        #         filepath='model.hdf5',
        #         monitor = "val_loss",
        #         verbose=1,
        #         save_best_only=True,
        #         mode = 'auto'
        #     ),
        #     tf.keras.callbacks.TensorBoard()
        # ]
    )
    model.summary()
    return model