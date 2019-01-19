from django.contrib import admin

from automl_systems.preprocessing.audio_to_spectogram_to_npy import transform_all_audio_files_to_npy
from training_server.models import AutoSklearnConfig

from automl_systems.auto_sklearn.run import train as train_auto_sklearn
from training_server.models.transform_audio_config import AudioReformater


class TransformAudioConfigAdmin(admin.ModelAdmin):
    list_display = ('status','folder_name', 'input_file_format', 'output_file_format')

    list_filter = ('status',)

    def get_readonly_fields(self, request, obj=None):
        readonly_fields = ['status', 'additional_remarks']
        return readonly_fields


    def save_model(self, request, obj, form, change):
        obj.save()
        # train_auto_sklearn.s(str(obj.id)).apply_async()
        transform_all_audio_files_to_npy(obj) # TODO Find out how to make async


admin.site.register(AudioReformater, TransformAudioConfigAdmin)
