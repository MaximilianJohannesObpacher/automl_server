from django.apps import AppConfig

class EvaluationConfig(AppConfig):
    name = 'evaluation'
    verbose_name = '3. Evaluation'

    # TODO comment this in to make the startup an experiment run.
    #def ready(self):
    #    from experiments_creator import start_experiment
    #    start_experiment()
