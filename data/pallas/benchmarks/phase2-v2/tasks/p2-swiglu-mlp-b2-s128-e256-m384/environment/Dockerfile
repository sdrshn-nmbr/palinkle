FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY starter/kernel.py /app/kernel.py
COPY public/dev_check.py /app/dev_check.py
COPY public/candidate_policy.py /app/candidate_policy.py
COPY public/PALLAS_API.md /app/PALLAS_API.md
RUN git init -q && git config user.name opjax-harness && git config user.email harness@opjax.invalid && git add . && git commit -q -m 'task base'
