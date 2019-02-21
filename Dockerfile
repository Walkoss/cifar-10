FROM tensorflow/tensorflow:latest-gpu-py3

RUN pip install talos

# talos include tensorflow remove the following lines for cpu usage
RUN pip uninstall -y \
        tensorflow \
        tensorflow-gpu

RUN pip install tensorflow-gpu
# <===> #