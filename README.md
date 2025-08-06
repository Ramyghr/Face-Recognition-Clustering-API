ENV=dev poetry run uvicorn app.main:app --reload

docker build --no-cache  -t facial-clustering:latest .

docker compose up

Missing Doccumentation (TBD)


resnet/vgg = (224,224) resize
facenet = (160,160)

steps to run locally

0° Python 3.12 installed

1° install poetry

`curl -sSL https://install.python-poetry.org | python -`

2° make sure CWD is where pyproject.toml is

3° run poetry install

`poetry install`

4° place models 

5° .env.dev file

6° run uvicorn:

`ENV=dev poetry run uvicorn app.main:app --reload`
