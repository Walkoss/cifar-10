import time
import tensorflow as tf

from models import BaseModel


class LinearModel(BaseModel):
    @classmethod
    def variant_default(cls, x_train, y_train, x_val, y_val, params):
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=x_train.shape[1:]),
                tf.keras.layers.Dense(10, activation=params["output-activation"]),
            ]
        )

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
            batch_size=params["batch-size"],
            verbose=1,
            callbacks=[
                tf.keras.callbacks.TensorBoard(
                    "./logs/"
                    + "linear_default/"
                    + "-".join("=".join((str(k), str(v))) for k, v in params.items())
                    + "-ts={}".format(str(time.time()))
                )
            ],
        )

        return history, model
