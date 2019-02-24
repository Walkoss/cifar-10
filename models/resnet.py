import time
import tensorflow as tf

from models import BaseModel


class ResNetModel(BaseModel):
    @classmethod
    def variant_default(cls, x_train, y_train, x_val, y_val, params):
        return cls.variant_resnet50(x_train, y_train, x_val, y_val, params)

    @classmethod
    def variant_resnet50(cls, x_train, y_train, x_val, y_val, params):
        base_model = tf.keras.applications.resnet50.ResNet50(
            weights="imagenet", include_top=False
        )
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        for i in range(params["hidden_layers"]):
            x = tf.keras.layers.Dense(params["units"], activation=params["activation"])(
                x
            )

        output = tf.keras.layers.Dense(10, activation="softmax")(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=output)

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

    @classmethod
    def variant_inceptionv3(cls, x_train, y_train, x_val, y_val, params):
        base_model = tf.keras.applications.InceptionV3(
            weights="imagenet", include_top=False
        )
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        for i in range(params["hidden_layers"]):
            x = tf.keras.layers.Dense(params["units"], activation=params["activation"])(
                x
            )

        output = tf.keras.layers.Dense(10, activation="softmax")(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=output)

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
                    + "inceptionv3/"
                    + "-".join("=".join((str(k), str(v))) for k, v in params.items())
                    + "-ts={}".format(str(time.time()))
                )
            ],
        )

        return history, model
