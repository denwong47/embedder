FROM python:3.11-slim-bookworm AS base

# Install dependencies
RUN apt update && apt install -y build-essential libtorch-dev curl pkg-config libssl-dev git

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | bash -s -- -y

# Set up environment for Cargo
RUN echo 'source $HOME/.cargo/env' >> $HOME/.bashrc

# Copy the Rust-Bert crate
WORKDIR /root/crates
RUN git clone https://github.com/guillaume-be/rust-bert.git
RUN pip install --upgrade pip && pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /root/crates/rust-bert/utils/requirements.txt

VOLUME ["/root/models"]

#
COPY docker/convert_model.sh /root
WORKDIR /root/models

ENTRYPOINT [ "/bin/bash", "/root/convert_model.sh" ]
