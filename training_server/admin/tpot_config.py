from django.contrib import admin

from training_server.models import AutoSklearnConfig, TpotConfig

#from automl_server.automl_systems.tpot.run import train as train_tpot


class TpotConfigAdmin(admin.ModelAdmin):

    fieldsets = (
        ('General Info:', {'fields': ('framework', 'status', 'date_trained', 'model_path')}),
        ('Resource Options:', {'fields': ('n_jobs', 'max_time_mins', 'max_eval_time_mins',)}),
        ('Model Training Options:', {'fields': (
        'generations', 'population_size', 'offspring_size', 'mutation_rate', 'crossover_rate', 'subsample', 'random_state', 'config_dict', 'warm_start', 'use_dask', 'early_stop', 'verbosity')}),
        ('Evaluation', {'fields': ('scoring', 'cv', )}),
        ('Caching and storage:', {'fields': (
        'memory',)})
    )

    def get_readonly_fields(self, request, obj=None):
        readonly_fields = ['status', 'model_path', 'date_trained', 'additional_remarks']
        if obj:
            if not 'framework' in readonly_fields:
                readonly_fields.append('framework')
            if obj.training_triggered:
                return [f.name for f in self.model._meta.fields]
        return readonly_fields


    def save_model(self, request, obj, form, change):
        obj.training_triggered = True
        #train_tpot(obj)
        obj.status = ('in_progress')
        super(TpotConfigAdmin, self).save_model(request, obj, form, change)

    def has_add_permission(self, request, obj=None):
        return False



admin.site.register(TpotConfig, TpotConfigAdmin)
