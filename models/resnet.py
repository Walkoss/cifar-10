import time
import tensorflow as tf
from tensorflow.python.keras import Input

from models import BaseModel


def resnet_layer(
    inputs,
    num_filters=16,
    kernel_size=3,
    strides=1,
    activation="relu",
    batch_normalization=True,
    conv_first=True,
):
    conv = tf.keras.layers.Conv2D(
        num_filters, kernel_size=kernel_size, strides=strides, padding="same"
    )

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x)
        if activation is not None:
            x = tf.keras.layers.Activation(activation)(x)
    else:
        if batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x)
        if activation is not None:
            x = tf.keras.layers.Activation(activation)(x)
        x = conv(x)
    return x


class ResNetModel(BaseModel):
    @classmethod
    def variant_default(cls, x_train, y_train, x_val, y_val, params):
        if (params["depth"] - 2) % 6 != 0:
            raise ValueError("depth should be 6n+2 (eg 20, 32, 44 in [a])")
        num_filters = 16
        num_res_blocks = int((params["depth"] - 2) / 6)

        inputs = Input(shape=x_train.shape[1:])
        x = resnet_layer(inputs=inputs)
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:
                    strides = 2
                y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
                y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
                if stack > 0 and res_block == 0:
                    x = resnet_layer(
                        inputs=x,
                        num_filters=num_filters,
                        kernel_size=1,
                        strides=strides,
                        activation=None,
                        batch_normalization=False,
                    )
                x = tf.keras.layers.add([x, y])
                x = tf.keras.layers.Activation(params["activation"])(x)
            num_filters *= 2

        x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
        y = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(10, activation=params["output_activation"])(y)

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
