FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src \
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 trafficpulse \
    && useradd --uid 10001 --gid trafficpulse --no-create-home trafficpulse

COPY pyproject.toml /tmp/pyproject.toml
COPY configs/phase6a-environment.txt /tmp/environment.txt
COPY docker/backend/container.py /opt/trafficpulse/container.py
RUN python /opt/trafficpulse/container.py requirements /tmp/pyproject.toml /tmp/environment.txt > /tmp/serving.txt \
    && pip install --no-cache-dir --only-binary=:all: -r /tmp/serving.txt \
    && rm /tmp/serving.txt /tmp/pyproject.toml /tmp/environment.txt

WORKDIR /app
COPY src/trafficpulse/ /app/src/trafficpulse/
COPY configs/reviewed_synthetic_artifacts.json /app/configs/reviewed_synthetic_artifacts.json
USER 10001:10001
EXPOSE 8000
ENTRYPOINT ["python", "/opt/trafficpulse/container.py", "serve"]
