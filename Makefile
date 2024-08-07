MODEL:=all-mpnet-base-v2
convert-model:
	docker compose build convert-model
	docker compose run -e MODEL=$(MODEL) convert-model
