training_server
===============

Prerequisites: Docker, Docker-compose

run with

'''
docker-compose up --build
'''

'''
celery -A training_server.celery worker -l DEBUG -E
'''
Install the redis server to run redis locally:

brew install redis

'''
Important:
'''

Do not forget to give docker a sufficient amount of resources for executign the tasks.

This is tested with 10 GiB and 2 Cores

Rel: https/github.com/docker/for-mac/issues/676#issuecomment-375901898