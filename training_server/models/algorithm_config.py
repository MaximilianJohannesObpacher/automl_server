from celery import Celery
from django.contrib.postgres.fields import ArrayField
from django.db import models

# from automl_server.automl_systems.autosklearn.run import train, add
# from automl_server.automl_systems.tpot.run import train as train_tpot


class AlgorithmConfig(models.Model):


    AUTO_SKLEARN = 'auto_sklearn'
    TPOT = 'tpot'

    ALGORITHM_CHOICES = (
        (AUTO_SKLEARN, 'Auto-sklearn'),
        (TPOT, 'TPOT'),
    )

    IN_PROGRESS = 'in_progress'
    SUCCESS = 'success'
    FAIL = 'fail'
    WAITING = 'waiting'

    STATUS_CHOICES = (
        (WAITING, 'Waiting for thread'),
        (IN_PROGRESS, 'In progress'),
        (SUCCESS, 'Success'),
        (FAIL, 'Fail')
    )

    framework = models.CharField(max_length=24, choices=ALGORITHM_CHOICES)
    model_path = models.CharField(null=True, blank=True, help_text='Path to the model', max_length=256)
    status = models.CharField(null=True, blank=True, help_text='Status of the training', choices=STATUS_CHOICES, max_length=32)
    date_trained = models.DateTimeField(auto_now=True)
    training_triggered = models.BooleanField(default=False, help_text='Helper Flag for defining which config should be updateable (which one has not yet been trained)')
    additional_remarks = models.CharField(null=True, blank=True, max_length=2048, help_text='Additional Information about the training. E.g. Information about failed trainings are logged here in case a training fails!')
    # input_data_filename = models.Charfield(default='merged_folds')
    # TODO Exclude this flag in the admin. Define making all fields readonly depending on that flag.

    #def save(self, *args, **kwargs):
    #    #(train.s()).apply_async()
    #    (train_tpot.s()).apply_async()
    #    # train_tpot()
    #    return super(AlgorithmConfig, self).save(*args, **kwargs)



