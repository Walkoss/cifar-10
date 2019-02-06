# nightly is used because there is a bug in latest tag
FROM tensorflow/tensorflow:nightly-gpu-py3-jupyter

RUN pip install \
        comet_ml \
        jupyterlab

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter lab --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root"]