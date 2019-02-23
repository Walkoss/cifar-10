import time
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Flatten, Dense, TimeDistributed, Dropout

from models import BaseModel


class LSTMModel(BaseModel):
    @classmethod
    def variant_default(cls, x_train, y_train, x_val, y_val, params):
        model = Sequential()

        model.add(TimeDistributed(Flatten(input_shape=(x_train.shape[1], x_train.shape[2]*x_train.shape[3]))))

        for i in range(params["lstm_layers"] - 1):
            model.add(LSTM(params["hidden_size"], return_sequences=True, activation=params["activation"]))

        model.add(LSTM(params["hidden_size"], activation=params["activation"]))

        if params["dropout"] > 0:
            model.add(Dropout(params["dropout"]))

        model.add(Dense(10, activation=params["output_activation"]))

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
        )

        return history, model
