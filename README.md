# cifar-10

## Requirements

These softwares must be installed on your local machine:

- Docker (with nvidia-docker2)

OR

- Conda virtual environment

## Installation

- Clone this repository and change directory to it
- Copy `env.example` as `.env` and edit environment variables
- Build custom image: `docker build -t tensorflow/tensorflow:custom-gpu-py3 .`

## Usage

- To start tensorflow container:

```bash
nvidia-docker run -it -u $(id -u):$(id -g) -v "$(realpath ./)":/tf/ -p 6006:6006 --name tf tensorflow/tensorflow:custom-gpu-py3 bash
```

From the container bash you can execute the train script:

```bash
cd tf
python train.py
# All available hyperparameters will be prompted.
```

Available models are `linear`, `mlp`, `cnn`, `resnet`, `lstm`

## Example

```bash
python train.py --model mlp --hidden-layers 2 6 8 --units 10 64 512 --activation relu sigmoid tanh --output-activation sigmoid softmax --lr 0.1 0.01 --epochs 10 50 100 --momentum 0.9 0.7 --batch-size 32 1024
```

It will train all model combinaison using Talos grid search. It will generate a csv file at the end.

## Optional

- You can execute `tensorboard`:

```bash
docker exec -d tf tensorboard --logdir=./logs
```

- Browse http://localhost:6006
