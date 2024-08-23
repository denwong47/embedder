FROM fedora:41 AS base

RUN dnf update -y && dnf group install -y "development-tools" && dnf install -y openssl-devel libstdc++-static
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > /tmp/rustup-init.sh && chmod u+x /tmp/rustup-init.sh && /tmp/rustup-init.sh -y
RUN echo 'source $HOME/.cargo/env' >> $HOME/.bashrc

WORKDIR /root/crate/embedder

COPY docker/build.sh /root/crate/build.sh

VOLUME ["/root/crate/embedder"]

FROM base AS build

ENTRYPOINT [ "sh", "-c", "/root/crate/build.sh" ]

FROM base AS local-build
ARG TARGET
ARG BUILD_ARGS
ENV TARGET=${TARGET}
ENV BUILD_ARGS=${BUILD_ARGS}

COPY . /root/crate/embedder

WORKDIR /root/crate/embedder
RUN MODEL_PATH=/root/crate/embedder/models TARGET=${TARGET} BUILD_ARGS=${BUILD_ARGS} /root/crate/build.sh

FROM base AS host

ARG TARGET
COPY --from=local-build /root/crate/embedder/target/${TARGET}/release/embedder /root/embedder

WORKDIR /root

ENTRYPOINT [ "/root/embedder" ]
