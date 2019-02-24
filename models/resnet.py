import time
import tensorflow as tf

from models import BaseModel


def _bn_relu(x):
    """Helper to build a BN -> relu block
    """
    norm = tf.keras.layers.BatchNormalization(axis=3)(x)
    return tf.keras.layers.Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu residual unit activation function.
       This is the original ResNet v1 scheme in https://arxiv.org/abs/1512.03385
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    padding = conv_params.setdefault("padding", "same")

    def f(x):
        x = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_size, strides=strides, padding=padding
        )(x)
        return _bn_relu(x)

    return f


def _shortcut(input_feature, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = tf.keras.backend.int_shape(input_feature)
    residual_shape = tf.keras.backend.int_shape(residual)
    stride_width = int(round(input_shape[1] / residual_shape[1]))
    stride_height = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]

    shortcut = input_feature
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = tf.keras.layers.Conv2D(
            filters=residual_shape[3],
            kernel_size=(1, 1),
            strides=(stride_width, stride_height),
            padding="valid",
        )(input_feature)
        shortcut = tf.keras.layers.BatchNormalization(axis=3)(shortcut)

    return tf.keras.layers.add([shortcut, residual])


def basic_block(
    filters,
    transition_strides=(1, 1),
    is_first_block_of_first_layer=False,
    dropout=None,
    residual_unit=_conv_bn_relu,
):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """

    def f(input_features):
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            x = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=(3, 3),
                strides=transition_strides,
                padding="same",
            )(input_features)
        else:
            x = residual_unit(
                filters=filters, kernel_size=(3, 3), strides=transition_strides
            )(input_features)

        if dropout is not None:
            x = tf.keras.layers.Dropout(dropout)(x)

        x = residual_unit(filters=filters, kernel_size=(3, 3))(x)

        return _shortcut(input_features, x)

    return f


def _residual_block(
    block_function,
    filters,
    blocks,
    transition_strides=None,
    is_first_layer=False,
    dropout=None,
    residual_unit=_conv_bn_relu,
):
    """Builds a residual block with repeating bottleneck blocks.
       stage: integer, current stage label, used for generating layer names
       blocks: number of blocks 'a','b'..., current block label, used for generating
            layer names
       transition_strides: a list of tuples for the strides of each transition
       transition_dilation_rates: a list of tuples for the dilation rate of each
            transition
    """
    if transition_strides is None:
        transition_strides = [(1, 1)] * blocks

    def f(x):
        for i in range(blocks):
            is_first_block = is_first_layer and i == 0
            x = block_function(
                filters=filters,
                transition_strides=transition_strides[i],
                is_first_block_of_first_layer=is_first_block,
                dropout=dropout,
                residual_unit=residual_unit,
            )(x)
        return x

    return f


def resnet(
    input_shape=None,
    classes=10,
    block="basic",
    repetitions=None,
    initial_filters=64,
    include_top=True,
    input_tensor=None,
    dropout=None,
    initial_strides=(2, 2),
    initial_kernel_size=(7, 7),
):
    if repetitions is None:
        repetitions = [3, 4, 6, 3]

    if block == "basic":
        block_fn = basic_block
    elif block == "bottleneck":
        raise NotImplementedError()

    img_input = tf.keras.layers.Input(shape=input_shape, tensor=input_tensor)
    x = _conv_bn_relu(
        filters=initial_filters,
        kernel_size=initial_kernel_size,
        strides=initial_strides,
    )(img_input)

    block = x
    filters = initial_filters
    for i, r in enumerate(repetitions):
        transition_strides = [(1, 1)] * r
        block = _residual_block(
            block_fn,
            filters=filters,
            blocks=r,
            is_first_layer=(i == 0),
            dropout=dropout,
            transition_strides=transition_strides,
            residual_unit=_conv_bn_relu,
        )(block)
        filters *= 2

    x = _bn_relu(block)

    if include_top:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(units=classes, activation="relu")(x)

    model = tf.keras.Model(inputs=img_input, outputs=x)
    return model


class ResNetModel(BaseModel):
    @classmethod
    def variant_default(cls, x_train, y_train, x_val, y_val, params):
        return cls.variant_resnet18(x_train, y_train, x_val, y_val, params)

    @classmethod
    def variant_resnet18(cls, x_train, y_train, x_val, y_val, params):
        model = resnet(
            input_shape=x_train.shape[1:],
            block="basic",
            repetitions=[2, 2, 2, 2],
            dropout=params["dropout"] > 0,
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
    def variant_resnet34(cls, x_train, y_train, x_val, y_val, params):
        model = resnet(
            input_shape=x_train.shape[1:],
            block="basic",
            repetitions=[3, 4, 6, 3],
            dropout=params["dropout"] > 0,
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
