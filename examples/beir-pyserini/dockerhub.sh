#!/bin/sh
#This tagname build the docker hub containers

# TAGNAME="1.0"

# docker build --no-cache -t beir/pyserini-fastapi:${TAGNAME} .
# docker push beir/pyserini-fastapi:${TAGNAME}

docker build --no-cache -t beir/pyserini-fastapi:latest .
docker push beir/pyserini-fastapi:latest