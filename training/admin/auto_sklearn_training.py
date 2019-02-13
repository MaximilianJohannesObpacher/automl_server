from django.contrib import admin
from django.core.exceptions import ValidationError
from django.http import HttpResponseRedirect
from django.shortcuts import redirect

from training.models import AutoSklearnTraining

from automl_systems.auto_sklearn.run import train as train_auto_sklearn


class AutoSklearnTrainingAdmin(admin.ModelAdmin):
    list_display = ('status', 'date_trained', 'model_path', 'additional_remarks')
    list_filter = ('status',)

    def get_fieldsets(self, request, obj=None):
        fieldsets = (
            ('General Info:', {'fields': (
            'training_name', 'framework', 'status', 'date_trained', 'model_path', 'logging_config',
            'additional_remarks', 'training_time')}),
            ('FileLoadingStrategy', {'fields': ('load_files_from',)}),
        )

        if not obj:
            return fieldsets
        else:
            fieldsets = list(fieldsets)
            fieldsets.append(['Resource Options:', {'fields': ('run_time', 'per_instance_runtime', 'memory_limit')}])
            fieldsets.append(['Model Training Options:', {'fields': ('initial_configurations_via_metalearning', 'ensemble_size', 'ensemble_nbest', 'seed',
                    'include_estimators',
                    'exclude_estimators', 'include_preprocessors', 'exclude_preprocessors', 'resampling_strategy',
                    'shared_mode')}])
            fieldsets.append(['Caching and storage:', {'fields': ('output_folder', 'delete_output_folder_after_terminate', 'tmp_folder',
                             'delete_tmp_folder_after_terminate', 'additional_remarks')}])


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
        readonly_fields = ['framework', 'status', 'model_path', 'date_trained', 'logging_config', 'additional_remarks', 'training_time']
        if obj:
            readonly_fields.append('load_files_from')
            if not 'framework' in readonly_fields:
                readonly_fields.append('framework')
            if obj.training_triggered and obj.freeze_results:
                return [f.name for f in self.model._meta.fields]
        return readonly_fields


    def save_model(self, request, obj, form, change):
        if obj.pk is None:
            obj.framework = 'auto_sklearn'
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
            train_auto_sklearn(str(obj.id)) # TODO Find out how to make async

    def response_add(self, request, obj, post_url_continue=None):
        return redirect('/admin/training/autosklearntraining/' + str(obj.id) + '/change/')


admin.site.register(AutoSklearnTraining, AutoSklearnTrainingAdmin)
