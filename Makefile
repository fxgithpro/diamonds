

setup : 
	pip install -e . 

launch-mlflow-server :
	echo "Launching MLflow server... "
	echo "All parameters are default in this command but you can customize them as needed."
	mlflow server --backend-store-uri sqlite:///models/mlflow.db --default-artifact-root ./models/mlartifacts --host 0.0.0.0 --port 5000