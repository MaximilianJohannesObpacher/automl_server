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

How to use the tool:

1. (Optional) If you want to transform png-images or wav-audiofiles to numpy array, put your files in the png/wav folder
and create an audio or png preprocessor. This will transform your data into numpy array suitable for machine learning.
If you already have the numpy arrays you want to use for machine learning, skip step one and put the files in the npy array.

2. Start training
First select your automl framework. (Auto-sklearn, tpot or auto-keras)
If you put your own numpy arrays in the numpy folder please select 'filenames' in the load files from section of your system.
Otherwise, please select Preprocessing Job output. Press save and in the next select the preprocessing job where you
preprocessed the data if you choose the preprocessing job output. otherwise set the filepath starting from the auto_ml data folder.
Afterwards set all parameters for your training and trigger it. Keep the tab active and prevent your computer from hibernating while the process is running

3. Evaluate your models
Select the model you created and the evaluation process you want to follow. run the evaluation


Warnings:
Check for the size of the input data. Big input data can easily exceed meemory limits of the container.

On a mac: Do not forget to give docker a sufficient amount of resources for executing the tasks.

This is tested with 10 GiB and 2 Cores

Rel: https/github.com/docker/for-mac/issues/676#issuecomment-375901898