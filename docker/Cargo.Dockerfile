ARG MODE=build
ARG TARGET=x86_64-unknown-linux-gnu

FROM rust:1-slim-bullseye AS cargo-base

ARG TARGET

ENV BUILD_TARGET=${TARGET}
ENV LIBTORCH_USE_PYTORCH=1

# Install dependencies
RUN apt update && apt install -y build-essential curl pkg-config libssl-dev git python3-pip
RUN python3 -m pip install --upgrade pip && python3 -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.1.0

RUN rustup component add rustfmt clippy

VOLUME ["/root/embedder"]
WORKDIR /root/embedder

FROM cargo-base AS cargo-build

ENTRYPOINT ["cargo", "build", "--target", "${BUILD_TARGET}"]

FROM cargo-base AS cargo-ci

ENTRYPOINT ["bash", "-c", "make", "cargo-ci"]

FROM cargo-${MODE} AS release
