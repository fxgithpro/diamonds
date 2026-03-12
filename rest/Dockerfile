FROM python:3.11-slim

# Do not start on the root directory
WORKDIR /app

# It is not bad to check for updates
RUN apt-get update

# Copy the diamond module and install it
COPY src src
COPY pyproject.toml pyproject.toml

ENV MODEL_FOLDER="models"

RUN python3.11 -m pip install --upgrade pip  
RUN pip install .

# Copy the REST API code and install its dependencies
COPY rest rest
RUN pip install -r rest/requirements.txt

# Do not do this in production, this is just for testing purposes
COPY models/preprocessor.pkl models/preprocessor.pkl
COPY models/model.pkl models/model.pkl

# Two way to launch a command with a variable
# CMD ["sh", "-c", "fastapi run rest/api.py --reload --port ${PORT}"]
# Maybe it would be also possible using the ENTRYPOINT instruction, but I am not sure how to do it with the fastapi command 
CMD fastapi run rest/api.py --reload --port ${PORT}