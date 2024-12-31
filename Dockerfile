    FROM python:3.10-slim

    WORKDIR /app

    RUN apt-get update && apt-get install -y \
        gcc \
        libmagic1 \
        build-essential \
        libssl-dev \
        libffi-dev \
        ffmpeg \
        libjpeg-dev \
        zlib1g-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        v4l-utils \
        tzdata \
            && ln -sf /usr/share/zoneinfo/America/Mexico_City /etc/localtime \
            && echo "America/Mexico_City" > /etc/timezone \
            && dpkg-reconfigure -f noninteractive tzdata \
            && apt-get clean

    ENV PYTHONDONTWRITEBYTECODE=1 PYTHONOPTIMIZE=1
    RUN python3 -m venv /venv
    ENV PATH="/venv/bin:$PATH"

    COPY ./requirements.txt /app/requirements.txt

    RUN pip install --no-cache-dir --upgrade pip

    RUN pip install --no-cache-dir --upgrade --compile -r /app/requirements.txt

    COPY . /app

    EXPOSE 8000
    #CMD ["fastapi", "run", "main:app", "--port", "8000"]