import time
import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Flatten
from models import BaseModel


class CNN(BaseModel):
    @classmethod
    def variant_default(cls, x_train, y_train, x_val, y_val, params):
        model = Sequential()
        model.add(Conv2D(params["filters"], (params["strides"], params["strides"]),
                         padding="same", input_shape=x_train.shape[1:]))
        model.add(Activation(params["activation"]))

        for i in range(params["layers"]):
            model.add(Conv2D(params["filters"], (params["strides"], params["strides"])))
            model.add(Activation(params["activation"]))
            if i != 0 and i % 2 == 0:
                model.add(MaxPooling2D(pool_size=(2, 2)))
                if params["dropout"] > 0:
                    model.add(Dropout(params["dropout"]))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10))
        model.add(Activation(params["output_activation"]))

        sgd = SGD(lr=params["lr"], decay=1e-6, nesterov=True)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        history = model.fit(
            x_train,
            y_train,
            batch_size=params["batch_size"],
            epochs=params["epochs"],
            validation_data=(x_val,y_val),
            shuffle=True,
            callbacks=[
                tf.keras.callbacks.TensorBoard(
                    "./logs/"
                    + "cnn_default/"
                    + "-".join("=".join((str(k), str(v))) for k, v in params.items())
                    + "-ts={}".format(str(time.time()))
                )
            ]
        )

        return history, model
