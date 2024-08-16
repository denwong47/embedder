FROM python:3.11-slim-bookworm AS base

# Install dependencies
RUN apt update && apt install -y build-essential
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu --upgrade pip optimum[exporters] sentence_transformers

VOLUME ["/root/models"]

RUN mkdir -p /root/models
WORKDIR /root/models
COPY docker/convert_model.sh /root/convert_model.sh

ENTRYPOINT [ "bash", "-c", "/root/convert_model.sh" ]
