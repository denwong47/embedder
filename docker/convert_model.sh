#!/usr/bin/env bash
if [ -d ./${MODEL} ]; then
    echo "Removing existing model directory."
    rm -rf ./${MODEL}
fi
optimum-cli export onnx --model ${MODEL} /tmp/output
if [ $? -ne 0 ]; then
    echo "Failed to convert model to ONNX."
    exit 1
fi
echo "Creating model directory."
mkdir -p ./${MODEL}
echo "Moving ONNX model to model directory."
mv /tmp/output/** ./${MODEL}
echo "Completed model conversion."
