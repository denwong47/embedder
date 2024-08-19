FROM fedora:41 AS base

RUN dnf update -y && dnf group install -y "development-tools" && dnf install -y openssl-devel libstdc++-static
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > /tmp/rustup-init.sh && chmod u+x /tmp/rustup-init.sh && /tmp/rustup-init.sh -y
RUN source $HOME/.cargo/env

VOLUME ["/root/crate/embedder"]

WORKDIR /root/crate/embedder

COPY docker/build.sh /root/crate/build.sh

ENTRYPOINT [ "sh", "-c", "/root/crate/build.sh" ]
