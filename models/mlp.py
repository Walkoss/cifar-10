import time
import tensorflow as tf

from models import BaseModel


class MLPModel(BaseModel):
    @classmethod
    def variant_default(cls, x_train, y_train, x_val, y_val, params):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=x_train.shape[1:]))

        for i in range(params["hidden-layers"]):
            model.add(
                tf.keras.layers.Dense(params["units"], activation=params["activation"])
            )

            if params["batch-norm"] > 0:
                model.add(tf.keras.layers.BatchNormalization())

            if params["dropout"] > 0:
                model.add(tf.keras.layers.Dropout(params["dropout"]))

        model.add(tf.keras.layers.Dense(10, activation=params["output_activation"]))

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
                    + "mlp_default/"
                    + "-".join("=".join((str(k), str(v))) for k, v in params.items())
                    + "-ts={}".format(str(time.time()))
                )
            ],
        )

        return history, model
