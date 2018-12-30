from django.contrib import admin

from training_server.models import AutoSklearnConfig

from automl_systems.auto_sklearn.run import train as train_auto_sklearn
# from automl_server.automl_systems.tpot.run import train as train_tpot


class AutoSklearnConfigAdmin(admin.ModelAdmin):
    list_display = ('status', 'date_trained', 'model_path', 'additional_remarks')

    fieldsets = (
        ('General Info:', {'fields':('framework', 'status', 'date_trained', 'model_path', 'logging_config', 'additional_remarks')}),
        ('Resource Options:', {'fields': ('run_time', 'per_instance_runtime', 'memory_limit')}),
        ('Model Training Options:', {'fields': ('initial_configurations_via_metalearning', 'ensemble_size', 'ensemble_nbest', 'seed', 'include_estimators', 'exclude_estimators', 'include_preprocessors', 'exclude_preprocessors', 'resampling_strategy', 'shared_mode')}),
        ('Caching and storage:', {'fields': ('output_folder', 'delete_output_folder_after_terminate', 'tmp_folder', 'delete_tmp_folder_after_terminate', 'additional_remarks')})
    )

    def get_readonly_fields(self, request, obj=None):
        readonly_fields = ['status', 'model_path', 'date_trained', 'logging_config', 'additional_remarks']
        if obj:
            if not 'framework' in readonly_fields:
                readonly_fields.append('framework')
            if obj.training_triggered:
                return [f.name for f in self.model._meta.fields]
        return readonly_fields


    def save_model(self, request, obj, form, change):
        obj.training_triggered = True
        obj.status = 'waiting'
        obj.save()
        train_auto_sklearn(obj)
        super(AutoSklearnConfigAdmin, self).save_model(request, obj, form, change)

    def has_add_permission(self, request, obj=None):
        return False


admin.site.register(AutoSklearnConfig, AutoSklearnConfigAdmin)
