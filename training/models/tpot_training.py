import datetime
import os
import time

import numpy
from django.db import models
from tpot import TPOTClassifier

from automl_server.settings import AUTO_ML_MODELS_PATH, AUTO_ML_DATA_PATH
from shared import reformat_data
from training.models.auto_ml_training import AutoMlTraining
import six.moves.cPickle as pickle


class TpotTraining(AutoMlTraining):
    ACCURACY = 'accuracy'
    ADJUSTED_RAND_SCORE = 'adjusted_rand_score'
    AVERAGE_PRECISION = 'average_precision'
    BALANCED_ACCURACY = 'balanced_accuracy'
    F1 = 'f1'
    F1_MACRO = 'f1_macro'
    F1_MICRO = 'f1_micro'
    F1_SAMPLES = 'f1_samples'
    F1_WEIGHTED = 'f1_weighted'
    NEG_LOG_LOSS = 'neg_log_loss'
    PRECISION = 'precision'
    PRECISION_MACRO = 'precision_macro'
    PRECISION_MICRO = 'precision_micro'
    PRECISION_SAMPLES = 'precision_samples'
    PRECISION_WEIGHTED = 'precision_weighted'
    RECALL = 'recall'
    RECALL_MACRO = 'recall_macro'
    RECALL_MICRO = 'recall_micro'
    RECALL_SAMPLES = 'recall_samples'
    RECALL_WEIGHTED = 'recall_weighted'
    ROC_AUC = 'roc_auc'


    SCORING_CHOICES = (
        (ACCURACY, 'accuracy'),
        (ADJUSTED_RAND_SCORE, 'adjusted_rand_score'),
        (AVERAGE_PRECISION, 'average_precision'),
        (BALANCED_ACCURACY, 'balanced_accuracy'),
        (F1, 'f1'),
        (F1_MACRO, 'f1_macro'),
        (F1_MICRO, 'f1_micro'),
        (F1_SAMPLES, 'f1_samples'),
        (F1_WEIGHTED, 'f1_weighted'),
        (NEG_LOG_LOSS, 'neg_log_loss'),
        (PRECISION, 'precision'),
        (PRECISION_MACRO, 'precision_macro'),
        (PRECISION_MICRO, 'precision_micro'),
        (PRECISION_SAMPLES, 'precision_samples'),
        (PRECISION_WEIGHTED, 'precision_weighted'),
        (RECALL, 'recall'),
        (RECALL_MACRO, 'recall_macro'),
        (RECALL_MICRO, 'recall_micro'),
        (RECALL_SAMPLES, 'recall_samples'),
        (RECALL_WEIGHTED, 'recall_weighted'),
        (ROC_AUC, 'roc_auc')
    )

    TPOT_LIGHT = 'TPOT light'
    TPOT_MDR = 'TPOT MDR'
    TPOT_SPARSE = 'TPOT sparse'

    CONFIG_CHOICES = (
        (TPOT_LIGHT, 'TPOT light'),
        (TPOT_MDR, 'TPOT MDR'),
        (TPOT_SPARSE, 'TPOT sparse')
    )

    AUTO = 'auto'
    NONE = None

    MEMORY_CHOICES = (
        (AUTO, 'auto'),
        (NONE, 'none')
    )

    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3

    VERBOSITY_CHOICES = (
        (ZERO, 0),
        (ONE, 1),
        (TWO, 2),
        (THREE, 3)
    )

    generations = models.IntegerField(default=100, null=True, blank=True, help_text='Number of iterations to the run pipeline optimization process. Must be a positive number. Generally, TPOT will work better when you give it more generations (and therefore time) to optimize the pipeline. TPOT will evaluate population_size + generations × offspring_size pipelines in total.')
    population_size = models.IntegerField(default=100, null=True, blank=True, help_text='Number of individuals to retain in the genetic programming population every generation. Must be a positive number. Generally, TPOT will work better when you give it more individuals with which to optimize the pipeline.')
    offspring_size = models.IntegerField(default=100, null=True, blank=True, help_text='Number of offspring to produce in each genetic programming generation. Must be a positive number.')
    mutation_rate = models.FloatField(default=0.9, null=True, blank=True, help_text='Mutation rate for the genetic programming algorithm in the range [0.0, 1.0]. This parameter tells the GP algorithm how many pipelines to apply random changes to every generation. mutation_rate + crossover_rate cannot exceed 1.0. We recommend using the default parameter unless you understand how the mutation rate affects GP algorithms.')
    crossover_rate = models.FloatField(default=0.1, null=True, blank=True, help_text='Crossover rate for the genetic programming algorithm in the range [0.0, 1.0]. This parameter tells the genetic programming algorithm how many pipelines to "breed" every generation. mutation_rate + crossover_rate cannot exceed 1.0. We recommend using the default parameter unless you understand how the crossover rate affects GP algorithms.')
    scoring = models.CharField(default=ACCURACY, null=True, blank=True, max_length=50, choices=SCORING_CHOICES, help_text='Function used to evaluate the quality of a given pipeline for the classification problem.')
    cv = models.IntegerField(default=5, null=True, blank=True, help_text='Specify the number of folds in a StratifiedKFold.')
    subsample = models.FloatField(default=1.0, null=True, blank=True, help_text='Fraction of training samples that are used during the TPOT optimization process. Must be in the range (0.0, 1.0].  Setting subsample=0.5 tells TPOT to use a random subsample of half of the training data. This subsample will remain the same during the entire pipeline optimization process.')
    n_jobs = models.IntegerField(default=1, null=True, blank=True, help_text='Number of processes to use in parallel for evaluating pipelines during the TPOT optimization process. Setting n_jobs=-1 will use as many cores as available on the computer. Beware that using multiple processes on the same machine may cause memory issues for large datasets')
    max_time_mins = models.IntegerField(default=None, null=True, blank=True, help_text='How many minutes TPOT has to optimize the pipeline. If not None, this setting will override the generations parameter and allow TPOT to run until max_time_mins minutes elapse.')
    max_eval_time_mins = models.IntegerField(default=None, null=True, blank=True, help_text='How many minutes TPOT has to evaluate a single pipeline. Setting this parameter to higher values will allow TPOT to evaluate more complex pipelines, but will also allow TPOT to run longer. Use this parameter to help prevent TPOT from wasting time on evaluating time-consuming pipelines.')
    random_state = models.IntegerField(default=None, null=True, blank=True, help_text='The seed of the pseudo random number generator used in TPOT. Use this parameter to make sure that TPOT will give you the same results each time you run it against the same data set with that seed.')
    config_dict = models.CharField(default=None, null=True, blank=True, max_length=50, choices=CONFIG_CHOICES, help_text="A configuration dictionary for customizing the operators and parameters that TPOT searches in the optimization process.\nPossible inputs are:\n string 'TPOT light', TPOT will use a built-in configuration with only fast models and preprocessors, or \nstring 'TPOT MDR', TPOT will use a built-in configuration specialized for genomic studies, or \nstring 'TPOT sparse': TPOT will use a configuration dictionary with a one-hot encoder and the operators normally included in TPOT that also support sparse matrices, or \nNone, TPOT will use the default TPOTClassifier configuration.")
    warm_start = models.NullBooleanField(default=False, null=True, blank=True, help_text='Flag indicating whether the TPOT instance will reuse the population from previous calls to fit(). Setting warm_start=True can be useful for running TPOT for a short time on a dataset, checking the results, then resuming the TPOT run from where it left off.')
    memory = models.CharField(default=None, null=True, blank=True, choices=MEMORY_CHOICES, max_length=50, help_text="String 'auto': TPOT uses memory caching with a temporary directory and cleans it up upon shutdown, or None, TPOT does not use memory caching.")
    use_dask = models.NullBooleanField(default=False, null=True, blank=True, help_text="Whether to use Dask-ML's pipeline optimiziations. This avoid re-fitting the same estimator on the same split of data multiple times. It will also provide more detailed diagnostics when using Dask's distributed scheduler.")
    # TODO Set periodic_checkpoint_folder
    early_stop = models.IntegerField(default=None, null=True, blank=True, help_text='How many generations TPOT checks whether there is no improvement in optimization process. Ends the optimization process if there is no improvement in the given number of generations.')
    verbosity = models.IntegerField(default=0, null=True, blank=True, choices=VERBOSITY_CHOICES, help_text="How much information TPOT communicates while it's running. \nPossible inputs are:\n0, TPOT will print nothing,\n1, TPOT will print minimal information,\n2, TPOT will print more information and provide a progress bar, or\n3, TPOT will print everything and provide a progress bar.")
    disable_update_check = models.NullBooleanField(default=False, null=True, blank=True, help_text='Flag indicating whether the TPOT version checker should be disabled. The update checker will tell you when a new version of TPOT has been released.')

    def train(self):
        print('in tpot training')
        try:
            # Storing save location for models
            dump_file = os.path.join(AUTO_ML_MODELS_PATH, 'tpot_' + str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')) + '.dump')

            x = numpy.load(os.path.join(AUTO_ML_DATA_PATH, self.training_data_filename))
            y = numpy.load(os.path.join(AUTO_ML_DATA_PATH, self.training_labels_filename))

            if self.preprocessing_object.input_data_type == 'png':
                x = reformat_data(x)

            # training the models
            print('about to train')
            model = TPOTClassifier(
                # verbosity=2, max_time_mins=90, max_eval_time_mins=5, config_dict='TPOT light', population_size=4, generations=3, n_jobs=1)
                generations=self.generations,
                population_size=self.population_size,
                offspring_size=self.offspring_size,
                mutation_rate=self.mutation_rate,
                crossover_rate=self.crossover_rate,
                scoring=self.scoring,
                cv=self.cv,
                subsample=self.subsample,
                n_jobs=self.n_jobs,
                max_time_mins=self.max_time_mins,
                # Tpot takes input in mins while most other frameworks take inputs in seconds.
                max_eval_time_mins=self.max_eval_time_mins,
                random_state=self.random_state,
                config_dict=self.config_dict,
                warm_start=self.warm_start,
                memory=self.memory,
                use_dask=self.use_dask,
                early_stop=self.early_stop,
                verbosity=self.verbosity,
                disable_update_check=self.disable_update_check
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

            self.training_time = round(end - start, 2)
            self.model_path = dump_file
            self.status = 'success'
            self.save()
            self.additional_remarks = str(model.fitted_pipeline_)
            self.save()

        except Exception as e:
            end = time.time()
            if 'start' in locals():
                self.training_time = round(end - start, 2)

            self.status = 'fail'
            self.additional_remarks = e
            self.save()
