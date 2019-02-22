import time
import tensorflow as tf

from models import BaseModel


class RNNModel(BaseModel):
    @classmethod
    def variant_default(cls, x_train, y_train, x_val, y_val, params):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=x_train.shape[1:]))

        for i in range(params["hidden_layers"]):
            model.add(
                tf.keras.layers.LSTM(params["units"], activation=params["activation"], return_sequences=True)
            )

            if params["batch_norm"] > 0:
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
            batch_size=params["batch_size"],
            verbose=1,
            callbacks=[
                tf.keras.callbacks.TensorBoard(
                    "./logs/"
                    + "rnn_default/"
                    + "-".join("=".join((str(k), str(v))) for k, v in params.items())
                    + "-ts={}".format(str(time.time()))
                )
            ],
        )

        return history, model
