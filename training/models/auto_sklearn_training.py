import datetime
import os
import time


import autosklearn.classification
import numpy
from django.contrib.postgres.fields import ArrayField
from django.db import models

from automl_server.settings import AUTO_ML_MODELS_PATH, AUTO_ML_DATA_PATH
from shared import reformat_data
from training.models.auto_ml_training import AutoMlTraining

import six.moves.cPickle as pickle

class AutoSklearnTraining(AutoMlTraining):
    RANDOM_FOREST = 'random_forest',
    ADABOOST = 'adaboost'
    BERNOULLI_NB = 'bernoulli_nb'
    DECISION_TREE = 'decission_tree'
    EXTRA_TREES = 'extra_trees'
    GAUSSIAN_NB = 'gaussian_nb'
    GRADIENT_BOOSTING = 'gradient_boosting'
    K_NEAREST_NEIGHBORS = 'k_nearest_neighbors'
    IDA = 'ida'
    LIBLINEAR_SVC = 'liblinear_svc'
    LIBSVM_SVC = 'libsvm_svc'
    MULTINOMIAL_NB = 'multinomial_nb'
    PASSIVE_AGGRESSIVE = 'passive_aggressive'
    QDA = 'qda'
    SGD = 'sgd'
    XGRADIENT_BOOSTING = 'xgradient_boosting'
    # TODO Add other

    CLASSIFICATION_ESTIMATORS = (
        (RANDOM_FOREST, 'random_forest'),
        (ADABOOST, 'adaboost'),
        (BERNOULLI_NB, 'bernoulli_nb'),
        (DECISION_TREE, 'decission_tree'),
        (EXTRA_TREES, 'extra_trees'),
        (GAUSSIAN_NB, 'guassian_nb'),
        (K_NEAREST_NEIGHBORS, 'k_nearest_neighbors'),
        (IDA, 'ida'),
        (LIBLINEAR_SVC, 'liblineaer_svc'),
        (LIBSVM_SVC, 'libsvm_svc'),
        (MULTINOMIAL_NB, 'multinomial_nb'),
        (PASSIVE_AGGRESSIVE, 'passive_aggressive'),
        (QDA, 'qda'),
        (SGD, 'sgd'),
        (XGRADIENT_BOOSTING, 'xgradient_boosting')
    )

    ARD_REGRESSION = 'ard_regression'
    GAUSSIAN_PROCESS = 'gaussian_process'
    LIBLINEAR_SVR = 'liblinear_svr'
    LIBSVM_SVR = 'libsvm_svr'
    RIDGE_REGRESSION = 'ridge_regression'

    REGRESSION_ESTIMATORS = (
        (ADABOOST, 'adaboost'),
        (DECISION_TREE, 'decission_tree'),
        (EXTRA_TREES, 'extra_trees'),
        (GAUSSIAN_PROCESS, 'gaussian_process'),
        (GRADIENT_BOOSTING, 'gradient_boosting'),
        (K_NEAREST_NEIGHBORS, 'k_nearest_neighbors'),
        (LIBLINEAR_SVR, 'liblinear_svr'),
        (LIBSVM_SVR, 'libsvm_svr'),
        (RANDOM_FOREST, 'random_forest'),
        (RIDGE_REGRESSION, 'ridge_regression'),
        (SGD, 'sgd'),
        (XGRADIENT_BOOSTING, 'xgradient_boosting')
    )

    ALL_ESTIMATORS = (
        (RANDOM_FOREST, 'random_forest'),
        (ADABOOST, 'adaboost'),
        (BERNOULLI_NB, 'bernoulli_nb'),
        (DECISION_TREE, 'decission_tree'),
        (EXTRA_TREES, 'extra_trees'),
        (GAUSSIAN_NB, 'guassian_nb'),
        (GAUSSIAN_PROCESS, 'gaussian_process'),
        (GRADIENT_BOOSTING, 'gradient_boosting'),
        (K_NEAREST_NEIGHBORS, 'k_nearest_neighbors'),
        (IDA, 'ida'),
        (LIBLINEAR_SVR, 'liblinear_svr'),
        (LIBSVM_SVR, 'libsvm_svr'),
        (LIBLINEAR_SVC, 'liblineaer_svc'),
        (LIBSVM_SVC, 'libsvm_svc'),
        (MULTINOMIAL_NB, 'multinomial_nb'),
        (PASSIVE_AGGRESSIVE, 'passive_aggressive'),
        (RANDOM_FOREST, 'random_forest'),
        (RIDGE_REGRESSION, 'ridge_regression'),
        (QDA, 'qda'),
        (SGD, 'sgd'),
        (XGRADIENT_BOOSTING, 'xgradient_boosting'))

    BALANCING = 'balancing'
    IMPUTATION = 'imputation'
    ONE_HOT_ENCODING = 'one_hot_encoding'
    RESCALING = 'rescalling'
    VARIANCE = 'variance'

    DATA_PREPROCESSORS = (
        (BALANCING, 'balancing'),
        (IMPUTATION, 'imputation'),
        (ONE_HOT_ENCODING, 'one_hot_encoding'),
        (RESCALING, 'rescalling'),
        (VARIANCE, 'variance')
    )

    DENSIFIER = 'densifier'
    EXTRA_TREES_PREPROC_FOR_CLASSIFICATION = 'extra_trees_preproc_for_classification'
    EXTRA_TREES_PREPROC_FOR_REGRESSION = 'extra_trees_preproc_for_regression'
    FAST_ICA = 'fast_ica'
    FEATURE_AGGLOMERATION = 'feature_agglomeration'
    KERNEL_PCA = 'kernel_pca'
    KITCHEN_SINKS = 'kitchen_sinks'
    LIBLINEAR_SVC_PREPROCESSORS = 'liblinear_svc_preprocessors'
    NO_PREPROCESSING = 'no_preprocessing'
    NYSTROEM_SAMPLER = 'nystroem_sampler'
    PCA = 'pca'
    POLYNOMIAL = 'polynomial'
    RANDOM_TREES_EMBEDDING = 'random_trees_embedding'
    SELECT_PERCENTILE = 'select_percentile'
    SELECT_PERCENTILE_CLASSIFICATION = 'select_percentile_classification'
    SELECT_PERCENTILE_REGRESSION = 'select_percentile_regression'
    SELECT_RATES = 'select_rates'
    TRUNCATEDSVD = 'truncatedsvd'

    FEATURE_PREPROCESSORS = (
        (DENSIFIER, 'densifier'),
        (EXTRA_TREES_PREPROC_FOR_CLASSIFICATION, 'extra_trees_preproc_for_classification'),
        (EXTRA_TREES_PREPROC_FOR_REGRESSION, 'extra_trees_preproc_for_regression'),
        (FAST_ICA, 'fast_ica'),
        (FEATURE_AGGLOMERATION, 'feature_agglomeration'),
        (KERNEL_PCA, 'kernel_pca'),
        (KITCHEN_SINKS, 'kitchen_sinks'),
        (LIBLINEAR_SVC_PREPROCESSORS, 'liblinear_svc_preprocessors'),
        (NO_PREPROCESSING, 'no_preprocessing'),
        (NYSTROEM_SAMPLER, 'nystroem_sampler'),
        (PCA, 'pca'),
        (POLYNOMIAL, 'polynomial'),
        (RANDOM_TREES_EMBEDDING, 'random_trees_embedding'),
        (SELECT_PERCENTILE, 'select_percentile'),
        (SELECT_PERCENTILE_CLASSIFICATION, 'select_percentile_classification'),
        (SELECT_PERCENTILE_REGRESSION, 'select_percentile_regression'),
        (SELECT_RATES, 'select_rates'),
        (TRUNCATEDSVD, 'truncatedsvd')
    )

    ALL_PREPROCESSORS = (
        (BALANCING, 'balancing'),
        (IMPUTATION, 'imputation'),
        (ONE_HOT_ENCODING, 'one_hot_encoding'),
        (RESCALING, 'rescalling'),
        (VARIANCE, 'variance'),
        (DENSIFIER, 'densifier'),
        (EXTRA_TREES_PREPROC_FOR_CLASSIFICATION, 'extra_trees_preproc_for_classification'),
        (EXTRA_TREES_PREPROC_FOR_REGRESSION, 'extra_trees_preproc_for_regression'),
        (FAST_ICA, 'fast_ica'),
        (FEATURE_AGGLOMERATION, 'feature_agglomeration'),
        (KERNEL_PCA, 'kernel_pca'),
        (KITCHEN_SINKS, 'kitchen_sinks'),
        (LIBLINEAR_SVC_PREPROCESSORS, 'liblinear_svc_preprocessors'),
        (NO_PREPROCESSING, 'no_preprocessing'),
        (NYSTROEM_SAMPLER, 'nystroem_sampler'),
        (PCA, 'pca'),
        (POLYNOMIAL, 'polynomial'),
        (RANDOM_TREES_EMBEDDING, 'random_trees_embedding'),
        (SELECT_PERCENTILE, 'select_percentile'),
        (SELECT_PERCENTILE_CLASSIFICATION, 'select_percentile_classification'),
        (SELECT_PERCENTILE_REGRESSION, 'select_percentile_regression'),
        (SELECT_RATES, 'select_rates'),
        (TRUNCATEDSVD, 'truncatedsvd')
    )

    AUTO_SKLEARN = 'Auto-sklearn'
    TPOT = 'TPOT'

    ALGORITHM_CHOICES = (
        (AUTO_SKLEARN, 'auto_sklearn'),
        (TPOT, 'TPOT'),
    )

    CLASSIFICATION = 'CL'
    REGRESSION = 'RG'
    ALL = 'ALL'

    TASK_CHOICES = (
        (REGRESSION, 'RG'),
        (CLASSIFICATION, 'CL'),
        (ALL, 'ALL')
    )

    IN_PROGRESS = 'in_progress'
    SUCCESS = 'success'
    FAIL = 'fail'

    STATUS_CHOICES = (
        (IN_PROGRESS, 'In progress'),
        (SUCCESS, 'Success'),
        (FAIL, 'Fail')
    )

    HOLDOUT = 'holdout'
    HOLDOUT_ITERATIVE_FIT = 'holdout_iterative_fit'
    CV = 'cv'
    PARTIAL_CV = 'partial_cv'

    RESAMPLING_STRATEGY_CHOICES = (
        (HOLDOUT, 'holdout'),
        (HOLDOUT_ITERATIVE_FIT, 'holdout_iterative_fit'),
        (CV, 'cv'),
        (PARTIAL_CV, 'partial_cv')
    )

    Y_OPTIMIZATION = 'y_optimization'
    MODEL = 'model' # TODO Maybe add string true here for passing it into the function and making bool field obsolete

    DISABLE_EVALUATOR_OUPUT_CHOICES = (
        (Y_OPTIMIZATION, 'y_optimization'),
        (MODEL, 'model')
    )

    run_time = models.IntegerField(default=3600, null=True, blank=True, help_text='Default: 3600. Time limit in seconds for the search of appropriate models. By increasing this value, the system has a higher chance of finding better models.!')
    per_instance_runtime = models.IntegerField(default=360, null=True, blank=True, help_text='Default: 360. Time limit for a single call to the machine learning model. Model fitting will be terminated if the machine learning algorithm runs over the time limit. Set this value high enough so that typical machine learning algorithms can be fit on the training data.')
    initial_configurations_via_metalearning = models.IntegerField(default=25, null=True, blank=True, help_text='Default: 25. Initialize the hyperparameter optimization algorithm with this many configurations which worked well on previously seen datasets. Disable if the hyperparameter optimization algorithm should start from scratch.')
    memory_limit = models.IntegerField(default=3072, null=True, blank=True, help_text='Default: 3072, Memory Limit for the Training.')
    ensemble_size = models.IntegerField(default=50, null=True, blank=True, help_text='Default: 50, Number of models added to the ensemble built by Ensemble selection from libraries of models. Models are drawn with replacement.')
    ensemble_nbest = models.IntegerField(default=50, null=True, blank=True, help_text='Default: 1, Only consider the ensemble_nbest models when building an ensemble. Implements Model Library Pruning from Getting the most out of ensemble selection.')
    seed = models.IntegerField(default=1, null=True, blank=True, help_text='Default: 1, Only consider the ensemble_nbest models when building an ensemble. Implements Model Library Pruning from Getting the most out of ensemble selection.')
    include_estimators = ArrayField(default=None, null=True, blank=True, choices=ALL_ESTIMATORS, base_field=models.CharField(max_length=64, null=True, blank=True), size=50, help_text='Default: None, If None, all possible estimators are used. Otherwise specifies set of estimators to use.')
    exclude_estimators = ArrayField(default=None, null=True, blank=True, choices=ALL_ESTIMATORS, base_field=models.CharField(max_length=64, null=True, blank=True), size=50, help_text='Default: None, If None, all possible estimators are used. Otherwise specifies set of estimators not to use. Incompatible with include_estimators.')
    include_preprocessors = ArrayField(default=None, null=True, blank=True, choices=ALL_PREPROCESSORS, base_field=models.CharField(max_length=64, null=True, blank=True), size=50, help_text='Default: None, If None all possible preprocessors are used. Otherwise specifies set of preprocessors to use.')
    exclude_preprocessors = ArrayField(default=None, null=True, blank=True, choices=ALL_PREPROCESSORS, base_field=models.CharField(max_length=64, null=True, blank=True), size=50, help_text='Default: None, If None all possible preprocessors are used. Otherwise specifies set of preprocessors not to use. Incompatible with include_preprocessors.')
    resampling_strategy = models.CharField(default=HOLDOUT, null=True, blank=True, choices=RESAMPLING_STRATEGY_CHOICES, max_length=128, help_text="Default: Holdout, how to to handle overfitting, might need ‘resampling_strategy_arguments’; Available arguments: ‘holdout’: {‘train_size’: float};‘holdout-iterative-fit’: {‘train_size’: float} ‘cv’: {‘folds’: int}‘partial-cv’: {‘folds’: int, ‘shuffle’: bool}")
    tmp_folder = models.CharField(default=None, null=True, blank=True, max_length=256, help_text='Default: None, folder to store configuration output and log files, if None automatically use /tmp/autosklearn_tmp_$pid_$random_number')
    output_folder = models.CharField(default=None, null=True, blank=True, max_length=256, help_text='Default: None, folder to store predictions for optional test set, if None automatically use , if None automatically use /tmp/autosklearn_output_$pid_$random_number')
    delete_tmp_folder_after_terminate = models.NullBooleanField(default=True, null=True, blank=True, help_text='Default: True, remove tmp_folder, when finished. If tmp_folder is None tmp_dir will always be deleted')
    delete_output_folder_after_terminate = models.NullBooleanField(default=True, null=True, blank=True, help_text='Default: True, remove output_folder, when finished. If output_folder is None output_dir will always be deleted')
    shared_mode = models.NullBooleanField(default=False, null=True, blank=True, help_text='Default: False, Run smac in shared-model-node. This only works if arguments tmp_folder and output_folder are given and both delete_tmp_folder_after_terminate and delete_output_folder_after_terminate are set to False.')
    # disable_evaluator_output = ArrayField(default=False, null=True, blank=True, choices=DISABLE_EVALUATOR_OUPUT_CHOICES, base_field=models.CharField(max_length=64), size=10, help_text='Can be used as a list to pass more fine-grained information on what to save. Allowed elements in the list are:')
    smac_scenario_args = models.CharField(default=None, null=True, blank=True, max_length=1024, help_text='Default: None, Additional arguments inserted into the scenario of SMAC. See the SMAC documentation for a list of available arguments.')
    # missing get_smac_object_callback - to advanced
    logging_config = models.CharField(null=True, blank=True, max_length=1024, help_text='dictionary object specifying the logger configuration. If None, the default logging.yaml file is used, which can be found in the directory util/logging.yaml relative to the installation.')

    # TODO Disable Save if training already triggered
    # TODO Enable API-Endpoints
    # TODO Add prediction and prediction result models and endpoints

    def train(self):

        self.status = 'in_progress'
        self.save()
        # Storing save location for models

        try:

            dump_file = os.path.join(AUTO_ML_MODELS_PATH, 'auto_sklearn' + str(datetime.datetime.now()) + '.dump')

            x = numpy.load(os.path.join(AUTO_ML_DATA_PATH, self.training_data_filename))
            y = numpy.load(os.path.join(AUTO_ML_DATA_PATH, self.training_labels_filename))

            if self.preprocessing_object.input_data_type == 'png':
                x = reformat_data(x)

            model = autosklearn.classification.AutoSklearnClassifier(
                time_left_for_this_task=self.run_time,
                per_run_time_limit=self.per_instance_runtime,
                initial_configurations_via_metalearning=self.initial_configurations_via_metalearning,
                ml_memory_limit=self.memory_limit,
                ensemble_size=self.ensemble_size,
                ensemble_nbest=self.ensemble_nbest,
                seed=self.seed,
                include_estimators=self.include_estimators,
                exclude_estimators=self.exclude_estimators,
                include_preprocessors=self.include_preprocessors,
                exclude_preprocessors=self.exclude_preprocessors,
                resampling_strategy=self.resampling_strategy,
                tmp_folder=self.tmp_folder,
                output_folder=self.output_folder,
                delete_tmp_folder_after_terminate=self.delete_tmp_folder_after_terminate,
                delete_output_folder_after_terminate=self.delete_output_folder_after_terminate,
                shared_mode=self.shared_mode,
                smac_scenario_args=self.smac_scenario_args,
                logging_config=self.logging_config,
            )
            print('before training start')
            start = time.time()
            model.fit(x, y)
            end = time.time()
            print(model.show_models())
            # storing the best performer
            with open(dump_file, 'wb') as f:
                pickle.dump(model, f)

            self.training_time = round(end - start, 2)
            self.status = 'success'
            self.model_path = dump_file
            self.save()
            print('Status final ' + self.status)

        except Exception as e:
            end = time.time()
            if 'start' in locals():
                print('failed after:' + str(end - start))
                self.training_time = round(end - start, 2)

            self.status = 'fail'
            self.additional_remarks = e
            self.save()

