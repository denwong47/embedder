FROM fedora:41 AS base

RUN dnf update -y && dnf group install -y "development-tools" && dnf install -y openssl-devel libstdc++-static
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > /tmp/rustup-init.sh && chmod u+x /tmp/rustup-init.sh && /tmp/rustup-init.sh -y
RUN echo 'source $HOME/.cargo/env' >> $HOME/.bashrc

WORKDIR /root/crate/embedder

COPY docker/build.sh /root/crate/build.sh

FROM base AS build

VOLUME ["/root/crate/embedder"]
ENTRYPOINT [ "sh", "-c", "/root/crate/build.sh" ]

FROM build AS host
# You can only run this target after running the build target

# ENV MODEL_PATH=/root/crate/embedder/models
# ENV PKG_CONFIG_SYSROOT_DIR=/

# COPY . .
# RUN source $HOME/.bashrc && cargo build --release --features=sentence_transformers
# RUN mv target/release/embedder /root/embedder
# RUN rm -rf /root/crate

ARG TARGET

COPY target/${TARGET}/release/embedder /root/embedder
WORKDIR /root

ENTRYPOINT [ "/root/embedder" ]
