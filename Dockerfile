FROM python:slim
WORKDIR /musicRankRegression/
ARG PORT_BUILD=6000
ENV PORT=$PORT_BUILD
EXPOSE $PORT_BUILD
COPY requirements.txt .
RUN pip install -r requirements.txt
ENTRYPOINT bash