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
from automl_server.settings import AUTO_ML_DATA_PATH, AUTO_ML_MODELS_PATH
from training.models import TpotTraining

random.seed(67)

import numpy as np
np.random.seed(67)

import os

target_name = 'bernie'

def train(tpot_training):
    try:
        tpot_training = TpotTraining.objects.get(id=tpot_training)
        # Storing save location for models
        dump_file = os.path.join(AUTO_ML_MODELS_PATH, 'tpot_' + str(datetime.datetime.now()) + '.dump')

        x = numpy.load(os.path.join(AUTO_ML_DATA_PATH, tpot_training.training_data_filename))
        y = numpy.load(os.path.join(AUTO_ML_DATA_PATH, tpot_training.training_labels_filename))

        if tpot_training.preprocessing_object.input_data_type == 'png':
            x = reformat_data(x)

        # training the models
        print('about to train')
        model = TPOTClassifier( #verbosity=2, max_time_mins=90, max_eval_time_mins=5, config_dict='TPOT light', population_size=4, generations=3, n_jobs=1)
            generations=tpot_training.generations,
            population_size=tpot_training.population_size,
            offspring_size=tpot_training.offspring_size,
            mutation_rate=tpot_training.mutation_rate,
            crossover_rate=tpot_training.crossover_rate,
            scoring=tpot_training.scoring,
            cv=tpot_training.cv,
            subsample=tpot_training.subsample,
            n_jobs=tpot_training.n_jobs,
            max_time_mins=tpot_training.max_time_mins, # Tpot takes input in mins while most other frameworks take inputs in seconds.
            max_eval_time_mins=tpot_training.max_eval_time_mins,
            random_state=tpot_training.random_state,
            config_dict=tpot_training.config_dict,
            warm_start=tpot_training.warm_start,
            memory=tpot_training.memory,
            use_dask=tpot_training.use_dask,
            early_stop=tpot_training.early_stop,
            verbosity=tpot_training.verbosity,
            disable_update_check=tpot_training.disable_update_check
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

        tpot_training.training_time = round(end - start, 2)
        tpot_training.model_path = dump_file
        tpot_training.status = 'success'
        tpot_training.save()

    except Exception as e:
        end = time.time()
        if 'start' in locals():
            tpot_training.training_time = round(end - start, 2)

        tpot_training.status = 'fail'
        tpot_training.additional_remarks = e
        tpot_training.save()
