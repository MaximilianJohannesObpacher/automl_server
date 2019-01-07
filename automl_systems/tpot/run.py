from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import datetime

import numpy
import six.moves.cPickle as pickle

import random

from tpot import TPOTClassifier

from training_server.celery import app
from automl_server.settings import AUTO_ML_DATA_PATH, AUTO_ML_MODELS_PATH

random.seed(67)

import numpy as np
np.random.seed(67)

import os

target_name = 'bernie'

@app.task()
def train(tpot_config):
    try:
        # Storing save location for models
        dump_file = os.path.join(AUTO_ML_MODELS_PATH, 'tpot_' + str(datetime.datetime.now()) + '.dump')

        x = numpy.load(os.path.join(AUTO_ML_DATA_PATH, 'merged_folds_training_x.npy'))
        y = numpy.load(os.path.join(AUTO_ML_DATA_PATH, 'merged_folds_training_y.npy'))

        nsamples = len(x)
        d2_npy = x.reshape((nsamples, -1))

        # replacing one_hot_encoding with letters for each category.
        labels = []
        labels_replace = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        for label in y:
            i = 0
            while i < 10:
                if label[i] == 1:
                    labels.append(labels_replace[i])
                    break
                i+=1

        # training the models
        print('about to train')
        model = TPOTClassifier(verbosity=2, max_time_mins=480, max_eval_time_mins=15, config_dict='TPOT light')
        #    generations=tpot_config.generations,
        #    population_size=tpot_config.population_size,
        #    offspring_size=tpot_config.offspring_size,
        #    mutation_rate=tpot_config.mutation_rate,
        #    crossover_rate=tpot_config.crossover_rate,
        #    scoring=tpot_config.scoring,
        #    cv=tpot_config.cv,
        #    subsample=tpot_config.subsample,
        #    n_jobs=tpot_config.n_jobs,
        #    max_time_mins=tpot_config.max_time_mins,
        #    max_eval_time_mins=tpot_config.max_eval_time_mins,
        #    random_state=tpot_config.random_state,
        #    config_dict=tpot_config.config_dict,
        #    warm_start=tpot_config.warm_start,
        #    memory=tpot_config.memory,
        #    use_dask=tpot_config.use_dask,
        #    early_stop=tpot_config.early_stop,
        #    verbosity=tpot_config.verbosity,
        #    disable_update_check=tpot_config.disable_update_check
        #)
        model.fit(d2_npy, labels)
        print('training finnished')


        with open(dump_file, 'wb') as f:
            print('about to save!')
            pickle.dump(model.fitted_pipeline_, f)
            print('model saved')

        tpot_config.model_path = dump_file
        tpot_config.status = 'success'
        tpot_config.save()

    except Exception as e:
        tpot_config.status = 'fail'
        tpot_config.additional_remarks = e
        tpot_config.save()
