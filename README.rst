training_server
===============

Prerequisites: Docker, Docker-compose

Download docker from

Build with (careful, this deletes all previous images and the db, therefore this should only be executed once when you checked out the project first):

'''
docker-compose up --build
'''

Run with

'''
docker-compose up
'''

Entrypoint for testing is 0.0.0.0:8003/admin/

The user is always defined in the entrypoint script.
By default a user with username: admin and password test1234 is created for you to test


brew install redis

'''
Important:
'''

On a mac: Do not forget to give docker a sufficient amount of resources for executing the tasks.

This is tested with 10 GiB and 2 Cores

Rel: https/github.com/docker/for-mac/issues/676#issuecomment-375901898