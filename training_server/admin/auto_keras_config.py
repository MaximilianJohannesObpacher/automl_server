from django.contrib import admin

from training_server.models import AutoSklearnConfig

from automl_systems.auto_keras.run import train as train_auto_keras
from training_server.models.auto_keras_config import AutoKerasConfig


class AutoKerasConfigAdmin(admin.ModelAdmin):
    list_display = ('status', 'date_trained', 'model_path', 'additional_remarks')

    fieldsets = (
        ('General Info:', {'fields':('framework', 'status', 'date_trained', 'model_path', 'additional_remarks', 'training_time', 'verbose')}),
        ('Resource Options:', {'fields': ('time_limit',)}),
        ('Caching and storage:', {'fields': ('input_data_filename', 'labels_filename')})
    )

    def get_readonly_fields(self, request, obj=None):
        readonly_fields = ['status', 'model_path', 'date_trained', 'logging_config', 'additional_remarks', 'training_time']
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
        train_auto_keras(str(obj.id)) # TODO Find out how to make async

    def has_add_permission(self, request, obj=None):
        return False


admin.site.register(AutoKerasConfig, AutoKerasConfigAdmin)
