from django.contrib import admin

# Register your models here.
from experiment_administration.models.experiment_supervisor import ExperimentSupervisor

admin.site.register(ExperimentSupervisor)
