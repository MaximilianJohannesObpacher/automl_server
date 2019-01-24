from django.contrib import admin
from django.core.exceptions import ValidationError
from django.shortcuts import redirect

from training_server.models import AutoSklearnConfig

from automl_systems.auto_keras.run import train as train_auto_keras
from training_server.models.auto_keras_config import AutoKerasConfig


class AutoKerasConfigAdmin(admin.ModelAdmin):
    list_display = ('status', 'date_trained', 'model_path', 'additional_remarks')
    list_filter = ('status',)

    def get_fieldsets(self, request, obj=None):
        fieldsets = (
            ('General Info:', {'fields': (
            'training_name', 'status', 'date_trained', 'model_path', 'additional_remarks', 'training_time', 'verbose',
            'additional_remarks', 'training_time')}),
            ('FileLoadingStrategy', {'fields': ('load_files_from',)}),
        )

        if not obj:
            return fieldsets
        else:
            fieldsets = list(fieldsets)
            fieldsets.append(['Resource Options:', {'fields': ('time_limit',)}])

            if obj.load_files_from == 'filename':
                fieldsets.append(('Input Files', {'fields': ('training_data_filename', 'training_labels_filename','validation_data_filename', 'validation_labels_filename')}))
            elif obj.load_files_from == 'preprocessing_job':
                fieldsets.append(('Input Object', {'fields': ('preprocessing_object', 'task_type')}))
                if obj.training_triggered:
                    fieldsets.append(('Input Files', {'fields': (
                    'training_data_filename', 'training_labels_filename', 'validation_data_filename',
                    'validation_labels_filename')}))
            return fieldsets

    def get_readonly_fields(self, request, obj=None):
        readonly_fields = ['status', 'model_path', 'date_trained', 'logging_config', 'additional_remarks', 'training_time']
        if obj:
            if not 'framework' in readonly_fields:
                readonly_fields.append('framework')
            if obj.training_triggered and obj.freeze_results:
                return [f.name for f in self.model._meta.fields]
        return readonly_fields

    def save_model(self, request, obj, form, change):
        if obj.pk is None:
            obj.framework = 'auto_keras'
            obj.save()
        else:
            if obj.load_files_from == 'preprocessing_job':
                if not obj.preprocessing_object:
                    raise ValidationError('No preprocessing object selected!')

                obj.training_data_filename = obj.preprocessing_object.training_features_path
                obj.validation_data_filename = obj.preprocessing_object.evaluation_features_path

                if obj.task_type == 'binary_classification':
                    obj.training_labels_filename = obj.preprocessing_object.training_labels_path_binary
                    obj.validation_labels_filename = obj.preprocessing_object.evaluation_labels_path_binary
                else:
                    obj.training_labels_filename = obj.preprocessing_object.training_labels_path
                    obj.validation_labels_filename = obj.preprocessing_object.evaluation_labels_path

            print(str(obj.training_data_filename) + str(obj.validation_data_filename) + str(obj.training_labels_filename) + str(obj.validation_labels_filename))
            obj.training_triggered = True
            obj.status = 'waiting'
            obj.save()
            train_auto_keras(str(obj.id))

    def response_add(self, request, obj, post_url_continue=None):
        return redirect('/admin/training_server/autokerasconfig/' + str(obj.id) + '/change/')

admin.site.register(AutoKerasConfig, AutoKerasConfigAdmin)
