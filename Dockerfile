FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

RUN pip install \
        comet_ml \
        jupyterlab \
        talos

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter lab --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root"]