FROM tensorflow/tensorflow:latest-gpu-py3

RUN pip install \
        comet_ml \
        talos

# talos include tensorflow remove the following lines for cpu usage
RUN pip uninstall \
        tensorflow \
        tensorflow-gpu

RUN pip install tensorflow-gpu
# <===> #

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter lab --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root"] 