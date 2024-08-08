#!/usr/bin/env bash
cargo fmt --all && \
cargo clippy --target ${BUILD_TARGET} --fix --allow-staged --allow-dirty --all-targets --all-features -- -D warnings && \
cargo test --target ${BUILD_TARGET} --all-features --all-targets
