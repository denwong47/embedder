.PHONY: convert-model cargo-ci cargo-ci-docker

MODEL:=all-mpnet-base-v2

RED:="\\033[0\;31m\\033\[1m"
RESET:="\\033\[0m"

BUILD_TARGET:=x86_64-unknown-linux-gnu

convert-model:
	@if [ -z "$(MODEL)" ]; then echo "$(RED)Environment Variable MODEL is not set.$(RESET) Please set it to the model you want to convert, e.g. 'make convert-model MODEL=sentence-transformers/all-mpnet-base-v2'."; exit 1; fi
	docker compose build convert-model
	docker compose run -e MODEL=$(MODEL) convert-model
