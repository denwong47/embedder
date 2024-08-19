#!/usr/bin/env bash
source $HOME/.cargo/env
rustup target add ${TARGET}
cargo build --target ${TARGET} ${BUILD_ARGS}
