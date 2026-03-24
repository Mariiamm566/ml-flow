FROM python:3.10-slim

ARG RUN_ID
ENV RUN_ID=${RUN_ID}

WORKDIR /app
COPY . /app

RUN echo "Downloading model for MLflow Run ID: ${RUN_ID}"

CMD ["sh", "-c", "echo Model container started for Run ID: ${RUN_ID}"]
