import time
import tensorflow as tf
from tensorflow.python.keras import Input

from models import BaseModel


class ResNetModel(BaseModel):
    @classmethod
    def variant_default(cls, x_train, y_train, x_val, y_val, params):
        base_model = tf.keras.applications.ResNet50(
            weights="imagenet",
            include_top=False,
            input_tensor=Input(shape=x_train.shape[1:]),
        )
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        outputs = tf.keras.layers.Dense(10, activation=params["output_activation"])(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

        for layer in base_model.layers:
            layer.trainable = False

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
                    + "resnet50/"
                    + "-".join("=".join((str(k), str(v))) for k, v in params.items())
                    + "-ts={}".format(str(time.time()))
                )
            ],
        )

        return history, model
