#!/usr/bin/env bash
if [ -z "$MODEL" ]; then
    echo "MODEL environment variable is not set."
    exit 1
fi
if [ -z "$TASK" ]; then
    echo "TASK environment variable is not set, defaulting to 'sentence-similiarity'."
    export TASK="sentence-similarity"
fi
if [ -d ./${MODEL} ]; then
    echo "Removing existing model directory."
    rm -rf ./${MODEL}
fi
# Possible tasks:
# fill-mask, mask-generation, token-classification, feature-extraction, object-detection, semantic-segmentation, audio-xvector, image-classification, depth-estimation, audio-classification, multiple-choice, image-to-image, image-to-text, automatic-speech-recognition, sentence-similarity, zero-shot-object-detection, audio-frame-classification, masked-im, question-answering, text-classification, text2text-generation, image-segmentation, text-to-audio, zero-shot-image-classification, text-generation
optimum-cli export onnx --task ${TASK} --model ${MODEL} /tmp/output
if [ $? -ne 0 ]; then
    echo "Failed to convert model to ONNX."
    exit 1
fi
echo "Creating model directory."
mkdir -p ./${MODEL}
echo "Moving ONNX model to model directory."
mv /tmp/output/** ./${MODEL}
echo "Completed model conversion."
