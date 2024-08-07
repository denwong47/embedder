#!/usr/bin/env bash
source /root/.bashrc
python /root/crates/rust-bert/utils/convert_model.py /root/models/${MODEL}/pytorch_model.bin
