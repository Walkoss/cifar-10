FROM tensorflow/tensorflow:latest-gpu-py3

RUN pip install talos

RUN pip uninstall -y \
        tensorflow \
        tensorflow-gpu

RUN pip install tensorflow-gpu==1.13.0rc1