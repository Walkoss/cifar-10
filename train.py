import argparse
import talos as ta
import tensorflow as tf

from models.cnn import CNNModel
from models.linear import LinearModel
from models.lstm import LSTMModel
from models.mlp import MLPModel

DEFAULT_LEARNING_RATE = 0.01
DEFAULT_MOMENTUM = 0.9
DEFAULT_BATCH_NORM = 0
DEFAULT_DROPOUT = 0
DEFAULT_UNIT = 32
DEFAULT_HIDDEN_LAYERS = 1
DEFAULT_ACTIVATION_FUNCTION = "relu"
DEFAULT_OUTPUT_ACTIVATION_FUNCTION = "softmax"
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_CONV_MODULES = 2
DEFAULT_FILTERS = 32
DEFAULT_KERNEL_SIZE = 3
DEFAULT_LSTM_LAYERS = 2
DEFAULT_HIDDEN_SIZE = 32

MODELS = {"linear": LinearModel, "mlp": MLPModel, "cnn": CNNModel, "lstm": LSTMModel}
ALLOWED_ACTIVATIONS = ["softmax", "sigmoid", "tanh", "relu"]


def main():
    parser = argparse.ArgumentParser("Train models on cifar-10")

    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", default="default")
    parser.add_argument("--lr", nargs="+", type=float, default=[DEFAULT_LEARNING_RATE])
    parser.add_argument("--momentum", nargs="+", type=float, default=[DEFAULT_MOMENTUM])
    parser.add_argument(
        "--batch-norm",
        nargs="+",
        type=int,
        default=[DEFAULT_BATCH_NORM],
        choices=[0, 1],
    )
    parser.add_argument("--dropout", nargs="+", type=float, default=[DEFAULT_DROPOUT])
    parser.add_argument("--units", nargs="+", type=int, default=[DEFAULT_UNIT])
    parser.add_argument(
        "--hidden-layers", nargs="+", type=int, default=[DEFAULT_HIDDEN_LAYERS]
    )
    parser.add_argument(
        "--activation",
        nargs="+",
        type=str,
        default=[DEFAULT_ACTIVATION_FUNCTION],
        choices=ALLOWED_ACTIVATIONS,
    )
    parser.add_argument(
        "--output-activation",
        nargs="+",
        type=str,
        default=[DEFAULT_OUTPUT_ACTIVATION_FUNCTION],
        choices=ALLOWED_ACTIVATIONS,
    )
    parser.add_argument("--epochs", nargs="+", type=int, default=[DEFAULT_EPOCHS])
    parser.add_argument(
        "--batch-size", nargs="+", type=int, default=[DEFAULT_BATCH_SIZE]
    )
    parser.add_argument(
        "--conv-modules", nargs="+", type=int, default=[DEFAULT_CONV_MODULES]
    )
    parser.add_argument("--filters", nargs="+", type=int, default=[DEFAULT_FILTERS])
    parser.add_argument(
        "--kernel-size", nargs="+", type=int, default=[DEFAULT_KERNEL_SIZE]
    )
    parser.add_argument(
        "--lstm-layers", nargs="+", type=int, default=[DEFAULT_LSTM_LAYERS]
    )
    parser.add_argument(
        "--hidden-size", nargs="+", type=int, default=[DEFAULT_HIDDEN_SIZE]
    )

    args = parser.parse_args()

    assert args.model in MODELS, "Model '{}' doesn't exist".format(args.model)
    model = MODELS[args.model].variant(args.variant)
    experiment = "{}_{}".format(args.model, args.variant)

    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    params = {
        "lr": args.lr,
        "momentum": args.momentum,
        "activation": args.activation,
        "output-activation": args.output_activation,
        "epochs": args.epochs,
        "batch-size": args.batch_size,
    }

    if args.model == "mlp" or args.model == "cnn" or args.model == "lstm":
        params.update(
            {
                "units": args.units,
                "hidden-layers": args.hidden_layers,
                "dropout": args.dropout,
                "batch-norm": args.batch_norm,
            }
        )

        if args.model == "cnn":
            params.update(
                {
                    "conv-modules": args.conv_modules,
                    "filters": args.filters,
                    "kernel-size": args.kernel_size,
                }
            )
        if args.model == "lstm":
            params.update(
                {
                    "lstm-layers": args.lstm_layers,
                    "hidden-size": args.hidden_size
                }
            )

    ta.Scan(
        x=x_train,
        y=y_train,
        x_val=x_test,
        y_val=y_test,
        model=model,
        params=params,
        dataset_name="cifar-10",
        experiment_no=experiment,
        clear_tf_session=False,
    )


if __name__ == "__main__":
    main()
