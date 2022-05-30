FROM python:slim as python_libs
WORKDIR /python_libs
COPY requirements.txt .
RUN pip install -r requirements.txt

FROM python:slim
COPY --from=python_libs /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
ENTRYPOINT bash