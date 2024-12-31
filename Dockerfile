
FROM python:3.10-slim


WORKDIR /code


COPY ./requirements.txt /code/requirements.txt

RUN apt-get update \
    && apt-get install gcc -y \
    && apt-get clean

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


COPY . /code


#CMD ["fastapi", "run", "main:app", "--port", "8000"]