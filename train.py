import argparse
import talos as ta
import tensorflow as tf

from models.linear import LinearModel

DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_MOMENTUM = 0.9
DEFAULT_ACTIVATION_FUNCTION = "sigmoid"
DEFAULT_OUTPUT_ACTIVATION_FUNCTION = "softmax"

MODELS = {"linear": LinearModel}


def main():
    parser = argparse.ArgumentParser("Train models on cifar-10")

    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", default="default")
    parser.add_argument("--epochs", nargs="+", type=int, default=[DEFAULT_EPOCHS])
    parser.add_argument("--lr", nargs="+", type=float, default=[DEFAULT_LEARNING_RATE])
    parser.add_argument("--momentum", nargs="+", type=float, default=[DEFAULT_MOMENTUM])
    parser.add_argument(
        "--batch-size", nargs="+", type=int, default=[DEFAULT_BATCH_SIZE]
    )
    parser.add_argument(
        "--activation",
        nargs="+",
        type=str,
        default=[DEFAULT_ACTIVATION_FUNCTION],
        choices=["sigmoid", "tanh", "relu"],
    )
    parser.add_argument(
        "--output-activation",
        nargs="+",
        type=str,
        default=[DEFAULT_OUTPUT_ACTIVATION_FUNCTION],
        choices=["sigmoid", "softmax"],
    )

    args = parser.parse_args()

    assert args.model in MODELS, "Model '{}' doesn't exist".format(args.model)
    model = MODELS[args.model].variant(args.variant)
    experiment = "{}_{}".format(args.model, args.variant)

    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    params = {
        "output_activation": args.output_activation,
        "lr": args.lr,
        "momentum": args.momentum,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
    }

    ta.Scan(
        x=x_train,
        y=y_train,
        x_val=x_train,
        y_val=y_train,
        model=model,
        params=params,
        dataset_name="cifar-10",
        experiment_no=experiment,
        clear_tf_session=False,
    )


if __name__ == "__main__":
    main()
