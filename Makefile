.PHONY: convert-model cargo-ci cargo-ci-docker

MODEL:=all-mpnet-base-v2

RED:="\\033[0\;31m\\033\[1m"
RESET:="\\033\[0m"

BUILD_TARGET:=x86_64-unknown-linux-gnu

cargo-ci:
	cargo fmt --all -- --check
	cargo clippy --target ${BUILD_TARGET} --fix --allow-staged --allow-dirty --all-features -- -D warnings && \
	cargo test --target ${BUILD_TARGET} --all-features

cargo-ci-docker:
	docker compose build cargo-ci
	docker compose run -T --remove-orphans cargo-ci

# Convert model to `rust_model.ot`.
# Supply the model name as `MODEL` environment variable; e.g. `make convert-model MODEL=all-mpnet-base-v2`.
convert-model:
	@if [ -z "$(MODEL)" ]; then echo "$(RED)Environment Variable MODEL is not set.$(RESET) Please set it to the model you want to convert, e.g. 'make convert-model MODEL=all-mpnet-base-v2'."; exit 1; fi
	@if [ ! -d "models/$(MODEL)" ]; then echo "$(RED)Model '$$MODEL' not found.$(RESET) Please make sure the model is downloaded and available in the 'models' directory."; exit 1; fi
	@if [ ! -f "models/$(MODEL)/pytorch_model.bin" ]; then echo "$(RED)Model '$$MODEL' is missing the 'pytorch_model.bin' file.$(RESET) Please make sure the model is downloaded and available in the 'models' directory."; exit 1; fi
	@first_line=$$(head -n 1 "models/$(MODEL)/pytorch_model.bin") && \
		if [[ $$first_line =~ ^version ]]; then echo "$(RED)Model '$$MODEL' is a Git LFS pointer file.$(RESET) Please install Git LFS and reclone the repository."; exit 2; fi
	docker compose build convert-model
	docker compose run -e MODEL=$(MODEL) convert-model
