    FROM python:3.10-slim

    WORKDIR /app

    RUN apt-get update && apt-get install -y \
        gcc \
        ffmpeg \
        libavformat-dev \
        libavutil-dev \
        libswscale-dev \
        libavdevice-dev \
        libx264-dev \
        libx265-dev
        libmagic1 \
        build-essential \
        libssl-dev \
        libffi-dev \
        libjpeg-dev \
        zlib1g-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        v4l-utils \
        tzdata \
        # Add these new dependencies for HEVC support
        libx264-dev \
        libx265-dev \
        libvpx-dev \
        libopencv-dev \
        python3-opencv \
        # Additional codecs and streaming support
        libavcodec-dev \
        && ln -sf /usr/share/zoneinfo/America/Mexico_City /etc/localtime \
        && echo "America/Mexico_City" > /etc/timezone \
        && dpkg-reconfigure -f noninteractive tzdata \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

    ENV PYTHONDONTWRITEBYTECODE=1 PYTHONOPTIMIZE=1
    ENV OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp|analyzeduration;10000000|buffer_size;2048000"
    RUN python3 -m venv /venv
    ENV PATH="/venv/bin:$PATH"

    COPY ./requirements.txt /app/requirements.txt

    RUN pip install --no-cache-dir --upgrade pip

    RUN pip install --no-cache-dir --upgrade --compile -r /app/requirements.txt

    COPY . /app

    EXPOSE 3001
    #CMD ["fastapi", "run", "main:app", "--port", "8000"]