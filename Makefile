.PHONY: convert-model cargo-ci cargo-ci-docker

MODEL:=all-mpnet-base-v2

RED:="\\033[0\;31m\\033\[1m"
RESET:="\\033\[0m"


convert-model:
	@if [ -z "$(MODEL)" ]; then echo "$(RED)Environment Variable MODEL is not set.$(RESET) Please set it to the model you want to convert, e.g. 'make convert-model MODEL=sentence-transformers/all-mpnet-base-v2'."; exit 1; fi
	docker compose build convert-model
	docker compose run -e MODEL=$(MODEL) convert-model

build:
	docker compose build build-binary
	docker compose run build-binary
# BUILD_TARGET:=x86_64-unknown-linux-gnu
# build: TARGET=$(BUILD_TARGET)
# build:
# 	CROSS_CONTAINER_OPTS="--volume $(pwd)/models:/tmp/models" cross build --target $(TARGET) --release --features=sentence_transformers

host:
	docker compose build build-host
	docker compose up build-host

export_host:
	docker save denwong47/embedder-host:latest | gzip > docker/embedder-host.tar.gz
