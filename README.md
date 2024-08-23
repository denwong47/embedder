# Embeddings Generation API

![Rust CI](https://github.com/denwong47/embedder/actions/workflows/rust-CI.yml/badge.svg?branch=main)

An experiment to see if Rust can do embeddings any faster, especially in the context of a web server.

## Motivation

Microservices doing NLP commonly requires embeddings to transform text into a vector space. This is a computationally expensive operation, and it is often done in Python using the `pytorch` library.

The `pytorch` library is typically used for training models, which it excels at; however for the purpose of generating embeddings, it is terribly bloated in size, and requires a lot of dependencies not required for the task at hand. This can seriously impact the cold start latency of such microservices hosted on the cloud such as AWS Lambda.

The project aims to detach such operations into a separate API hosted on Kubernetes, allow all microservices requiring embeddings to share the scaling of this API, minimizing the probability of cold starts.

Currently, this project proposes to use Rust `tokio`, `axum` and `fastembed-rs`.

## How it works

The project is split into three parts:

- `embedder-lib`: A library that wraps the `fastembed-rs` library, which is a wrapper around the `ort` library. This library uses ONNX models to generate embeddings, and does not depend on `libtorch`.

  > [!TIP]
  > This library will embed the whole model into the binary, in order to simplify deployment. This lengthens the build time of the library, but makes the resulting binary more portable.

- `make convert_model MODEL=...`: A Make target that converts a PyTorch model into an ONNX model. This is required for the `embedder-lib` to work. This utilizes `torch` and `optimium` within a Docker container, which is not required for the final deployment.
- `make host`: A Make target that hosts the API on `localhost:3000`. This is a simple API that accepts a `POST` request with a JSON body containing a list of strings, and returns a JSON body containing a list of embeddings. This can be followed by `make export_host` to export the built image into a Docker image for deployment.

## How to run

Currently this project is built for x86 platforms only, but can be easily extended to ARM platforms by modifying the `docker-compose.yml` file.

Pre-requisites are only required within the build and host docker images, so as long as you have docker, you should be able to run this project.

An example workflow will be:

- Clone this repository.
- Run `make convert_model MODEL=...` to convert your PyTorch model into an ONNX model.
  - The `MODEL` variable should be the HuggingFace model name, e.g. `sentence-transformers/all-mpnet-base-v2` or `sentence-transformers/all-MiniLM-L6-v2`.
  - The models will be exported to the `models` directory.
- Run `make host` to host the API on `localhost:3000`.
- Test the endpoint using your desired HTTP client, such as `requests` in Python:

  ```python
  >>> import requests
  >>> import numpy as np
  >>> docs = [
  ...     "This is a test sentence.",
  ...     "This is another test sentence.",
  ...     "Foo Bar",
  ... ]
  >>> response = requests.post("http://localhost:3000/embed?output=array", json={ "model": "sentence-transformers/all-MiniLM-L6-v2", "documents": docs })
  >>> response.json()
  {'duration': 0.5497052669525146,
   'embeddings': {'data': [0.08429647982120514,
     0.05795367807149887,
     ...],
    'dim': [3, 384],
    'v': 1},
   'model': 'sentence-transformers/all-MiniLM-L6-v2'}
  >>> np.array(response.json()["embeddings"]["data"]).reshape(response.json()["embeddings"]["dim"])
  array([[ 0.08429648,  0.05795368,  0.00449334, ...,  0.00457119,
         0.08188032, -0.0990471 ],
       [ 0.07382309,  0.05248551, -0.00273446, ..., -0.00953856,
         0.07654411, -0.06636316],
       [-0.00179896, -0.01348714,  0.02420126, ...,  0.06645426,
         0.06204206,  0.06890305]])
  ```
- Run `make export_host` to export the built image into a Docker image for deployment; the image will be saved to `docker/embedder-host.tar.gz`.

> [!NOTE]
> Note that the image MUST be in Fedora, as the compilation requires new versions of `gcc` and `glibc` that are not available in the Debian images or Alpine images.


## Performance and observations

Since the `fastembed-rs` library is a wrapper around the `ort` library, it does not necessarily has an edge over any other languages using the `onnxruntime` library. Even if Rayon is used to parallelize the embeddings, the speedup is not significant due to the `onnxruntime` already using all the available cores.

It is observed that some batch sizes perform better than others. Ultimately, the best batch size is dependent on the model and the hardware it is running on, but it is possible that there exists a algorithmic way to determine the best batch size for a given model and hardware, which can be implemented in the future.

## For your sanity 🧹

> [!IMPORTANT]
> Remember to do a lot of `cargo clean`, particularly on `--package embedder-lib`.


## Future work

Exposes CUDA support from `ort` into the `fastembed-rs` library, which will allow the embeddings to be generated on the GPU. Test if such a compiled binary will need to carry all the Nvidia SDK dependencies, which was a nightmare to deploy in the past. (Python wheels demanding Nvidia SDK during docker build, which is not available on the Nvidia docker image - only on the host machine, which we don't control.)
