
FROM python:3.8-buster
# this dockerfile is used for product deployments

COPY . /app
WORKDIR /app

# COPY requirements.txt requirements.txt

# RUN apk update && \
#     apk add --virtual build-deps gcc musl-dev && \
#     apk add postgresql-dev && \
#     rm -rf /var/cache/apk/*

RUN pip install -r requirements.txt
# RUN apk del build-deps gcc musl-dev

# for the flask config
ENV FLASK_ENV=prod

EXPOSE 5000
ENTRYPOINT [ "gunicorn", "-b", "0.0.0.0:5000", "--log-level", "INFO", "wsgi:app" ]
