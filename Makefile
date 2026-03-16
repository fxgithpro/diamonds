

setup : 
	@echo "Install the module"
	pip install -e . 

train:
	@echo "Start training the model"
	python -m diamonds.train

launch-mlflow-server :
	@echo "Launching MLflow server... "
	@echo "All parameters are default in this command but you can customize them as needed."
	mlflow server --backend-store-uri sqlite:///models/mlflow.db --default-artifact-root ./models/mlartifacts --host 0.0.0.0 --port 5000

########################################################################################################################

# REST API

########################################################################################################################
start_fastapi:
	fastapi run rest/api.py --reload --port 8888

start_fastapi_dev:
	fastapi dev rest/api.py --reload --port 8888

test_price_api:
	curl -X 'POST' \
	'http://127.0.0.1:8888/price' \
	-H 'accept: application/json' \
	-H 'Content-Type: application/json' \
	-d '{ "carat": 0.25, "cut": "Ideal", "color": "E", "clarity": "SI2", "depth": 61.5, "table": 55.0, "x": 3.95, "y": 3.98, "z": 2.43 }'

########################################################################################################################

# Docker commands

########################################################################################################################
build_local:
	@echo "Build the docker ${IMAGE}"
	docker build -t ${IMAGE} --file rest/Dockerfile .

build_gcp:
# Build the image for GCP (Linux/amd64 platform required for Cloud Run)
	docker build --platform linux/amd64 -t ${LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE} --file rest/Dockerfile .

run_local: build_local 
# Run local image
	@echo "Run the docker image"
	docker run -p ${HOST_PORT}:${PORT} -e PORT=${PORT} ${IMAGE}


push_gcp: build_gcp
# Push the image to Artifact Registry
	docker push ${LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE}

auth_gcp:
	gcloud auth login
	gcloud config set project ${PROJECT_ID}

deploy:
	gcloud run deploy ${IMAGE} --image ${LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE} --region ${LOCATION} --platform managed --allow-unauthenticated