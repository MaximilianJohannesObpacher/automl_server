from django.contrib import admin
from django.core.exceptions import ValidationError
from django.http import HttpResponseRedirect
from django.shortcuts import redirect
from django.utils.safestring import mark_safe

from automl_server import settings
from training.models import TpotTraining


class TpotTrainingAdmin(admin.ModelAdmin):
    list_display = ('status', 'date_trained', 'model_link', 'additional_remarks')
    list_filter = ('status',)

    # TODO Maybe a mixin would be a elegant solution for this.

    def get_fieldsets(self, request, obj=None):
        fieldsets = (
            ('General Info:', {'fields': (
            'training_name', 'status', 'date_trained', 'model_link',
            'additional_remarks', 'training_time')}),
            ('File Loading Strategy', {'fields': ('load_files_from',)}),
        )

        if not obj:
            return fieldsets
        else:
            fieldsets = list(fieldsets)
            fieldsets.append(['Resource Options:', {'fields': ('n_jobs', 'max_time_mins', 'max_eval_time_mins',)}])
            fieldsets.append(['Model Training Options:', {'fields': (
        'generations', 'population_size', 'offspring_size', 'mutation_rate', 'crossover_rate', 'subsample',
        'random_state', 'config_dict', 'warm_start', 'use_dask', 'early_stop', 'verbosity')}])
            fieldsets.append(['Caching and storage:', {'fields': ('memory',)}])
            fieldsets.append(['Evaluation', {'fields': ('scoring', 'cv',)}])

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
        readonly_fields = ['status', 'model_link', 'date_trained', 'additional_remarks', 'training_time']
        if obj:
            readonly_fields.append('load_files_from')
            if not 'framework' in readonly_fields:
                readonly_fields.append('framework')
            if obj.training_triggered and obj.freeze_results:
                return [f.name for f in self.model._meta.fields]
        return readonly_fields

    def model_link(self, obj):
        return mark_safe('<a href="' + obj.model_path.replace('/code', settings.BASE_URL)+ '">'+ obj.model_path.replace('/code', settings.BASE_URL)+' </a>' )

    def save_model(self, request, obj, form, change):
        if obj.pk is None:
            obj.framework = 'tpot'
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
            obj.train()


    def response_add(self, request, obj, post_url_continue=None):
        return redirect('/admin/training/tpottraining/' + str(obj.id) + '/change/')

admin.site.register(TpotTraining, TpotTrainingAdmin)
