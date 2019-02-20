# cifar-10

## Requirements

These softwares must be installed on your local machine:

- Docker (with nvidia-docker2)

## Installation

-  Clone this repository and change directory to it
- Copy `env.example` as `.env` and edit environment variables
- Build custom image: `docker build -t tensorflow/tensorflow:custom-gpu-py3 .`

## Usage

- To start tensorflow container which runs jupyterlab in background:
```bash
nvidia-docker run -d -u $(id -u):$(id -g) -v "$(realpath ./notebooks)":/tf/ -p 8888:8888 -p 6006:6006 --env-file .env --name tf tensorflow/tensorflow:custom-gpu-py3
```
- Note the generated token:
```bash
docker logs tf
```
- Browse http://localhost:8888 and enter the token

## Optional

- You can execute `tensorboard`:

```bash
docker exec -d tf tensorboard --logdir=./logs
```

- Browse http://localhost:6006
