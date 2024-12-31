
FROM python:3.10-slim


WORKDIR /app


COPY ./requirements.txt /code/requirements.txt

RUN apt-get update \
    && apt-get install gcc -y \
    && apt-get clean

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt


COPY . /app


#CMD ["fastapi", "run", "main:app", "--port", "8000"]