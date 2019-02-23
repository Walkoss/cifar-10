import time
import tensorflow as tf
from tensorflow.python.keras import Input

from models import BaseModel


class ResNetModel(BaseModel):
    @classmethod
    def variant_default(cls, x_train, y_train, x_val, y_val, params):
        inputs = Input(shape=x_train.shape[1:])
        x = tf.keras.layers.Conv2D(
            params["filters"],
            (params["kernel_size"], params["kernel_size"]),
            padding="same",
        )(inputs)
        if params["batch_norm"] > 0:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation=params["activation"])(x)

        for i in range(params["residual_blocks"]):
            x1 = tf.keras.layers.Conv2D(
                params["filters"],
                (params["kernel_size"], params["kernel_size"]),
                padding="same",
            )(x)
            if params["batch_norm"] > 0:
                x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(activation=params["activation"])(x)
            x = tf.keras.layers.Add()([x, x1])

        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(10, activation=params["output_activation"])(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=tf.keras.optimizers.SGD(
                lr=params["lr"], momentum=params["momentum"]
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            verbose=1,
            callbacks=[
                tf.keras.callbacks.TensorBoard(
                    "./logs/"
                    + "resnet_default/"
                    + "-".join("=".join((str(k), str(v))) for k, v in params.items())
                    + "-ts={}".format(str(time.time()))
                )
            ],
        )

        return history, model
