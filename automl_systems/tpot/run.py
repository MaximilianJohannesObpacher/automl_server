from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import datetime
import math
import multiprocessing
import time

import numpy
import six.moves.cPickle as pickle

import random

from tpot import TPOTClassifier

from automl_systems.shared import load_ml_data, reformat_data
from training_server.celery import app
from automl_server.settings import AUTO_ML_DATA_PATH, AUTO_ML_MODELS_PATH
from training_server.models import TpotConfig

random.seed(67)

import numpy as np
np.random.seed(67)

import os

target_name = 'bernie'

@app.task()
def train(tpot_config):
    try:
        tpot_config = TpotConfig.objects.get(id=tpot_config)
        # Storing save location for models
        dump_file = os.path.join(AUTO_ML_MODELS_PATH, 'tpot_' + str(datetime.datetime.now()) + '.dump')

        x = numpy.load(os.path.join(AUTO_ML_DATA_PATH, tpot_config.training_data_filename))
        y = numpy.load(os.path.join(AUTO_ML_DATA_PATH, tpot_config.training_labels_filename))

        if tpot_config.preprocessing_object.input_data_type == 'png':
            x = reformat_data(x)

        # training the models
        print('about to train')
        model = TPOTClassifier( #verbosity=2, max_time_mins=90, max_eval_time_mins=5, config_dict='TPOT light', population_size=4, generations=3, n_jobs=1)
            generations=tpot_config.generations,
            population_size=tpot_config.population_size,
            offspring_size=tpot_config.offspring_size,
            mutation_rate=tpot_config.mutation_rate,
            crossover_rate=tpot_config.crossover_rate,
            scoring=tpot_config.scoring,
            cv=tpot_config.cv,
            subsample=tpot_config.subsample,
            n_jobs=tpot_config.n_jobs,
            max_time_mins=tpot_config.max_time_mins, # Tpot takes input in mins while most other frameworks take inputs in seconds.
            max_eval_time_mins=tpot_config.max_eval_time_mins,
            random_state=tpot_config.random_state,
            config_dict=tpot_config.config_dict,
            warm_start=tpot_config.warm_start,
            memory=tpot_config.memory,
            use_dask=tpot_config.use_dask,
            early_stop=tpot_config.early_stop,
            verbosity=tpot_config.verbosity,
            disable_update_check=tpot_config.disable_update_check
        )
        print('before training start')
        start = time.time()
        model.fit(x, y)
        end = time.time()
        print('training finnished')


        with open(dump_file, 'wb') as f:
            print('about to save!')
            pickle.dump(model.fitted_pipeline_, f)
            print('model saved')

        tpot_config.training_time = round(end - start, 2)
        tpot_config.model_path = dump_file
        tpot_config.status = 'success'
        tpot_config.save()

    except Exception as e:
        end = time.time()
        if 'start' in locals():
            tpot_config.training_time = round(end - start, 2)

        tpot_config.status = 'fail'
        tpot_config.additional_remarks = e
        tpot_config.save()
