FROM python:3.11

WORKDIR /tmp
COPY requirements.txt .

ENV PYTHONUNBUFFERED=True \
    PORT=${PORT:-9090} \
    PIP_CACHE_DIR=/.cache
RUN apt update && apt install build-essential -y \
    && apt install python3-dev -y

RUN --mount=type=cache,target=$PIP_CACHE_DIR \
    pip install -r requirements.txt
COPY uwsgi.ini /etc/uwsgi/
COPY supervisord.conf /etc/supervisor/conf.d/

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
WORKDIR /app

COPY *.py /app/
COPY *.pt /models/

COPY courtvision-0.1.0-py3-none-any.whl courtvision-0.1.0-py3-none-any.whl
RUN pip install courtvision-0.1.0-py3-none-any.whl
EXPOSE 9090
# CMD [ "echo $(which supervisord)" ]
CMD ["/usr/local/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
