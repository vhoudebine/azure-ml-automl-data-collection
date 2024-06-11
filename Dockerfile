FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04

#USER dockeruser

ENV CONDA_ENV_DIR=/opt/miniconda/envs
# python installs
COPY mlflow-model/conda.yaml /tmp/conda.yaml

RUN conda env create -n userenv -f /tmp/conda.yaml && \
    $CONDA_ENV_DIR/userenv/bin/pip install azureml-automl-runtime==1.56.0

# Update environment variables
ENV AZUREML_CONDA_ENVIRONMENT_PATH="$CONDA_ENV_DIR/userenv" 
ENV PATH="$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH" 
ENV LD_LIBRARY_PATH="$AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH"