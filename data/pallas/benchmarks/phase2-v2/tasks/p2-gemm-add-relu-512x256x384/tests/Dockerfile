FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7
RUN pip install --no-cache-dir jax==0.10.1 jaxlib==0.10.1 chex==0.1.91 numpy==2.2.6 ml_dtypes==0.5.3 scipy==1.15.3 tomli==2.2.1
COPY task.json /tests/task.json
COPY test.sh /tests/test.sh
COPY opjax /opt/opjax
RUN chmod +x /tests/test.sh
ENV PYTHONPATH=/opt
