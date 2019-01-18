from django.contrib import admin
from django.http import HttpResponseRedirect

from training_server.models import AlgorithmConfig, AutoSklearnConfig, TpotConfig, AutoKerasConfig


class AlgorithmConfigAdmin(admin.ModelAdmin):

    def get_readonly_fields(self, request, obj=None):
        return ('status', 'model_path', 'date_trained', 'training_triggered', 'additional_remarks', 'training_time')

    def save_model(self, request, obj, form, change):
        pass # Not saving the basemodel algorithmconfig, but instead the model autosklearn_config or tpot_config in the response add method


    # TODO refactor for one set
    # depending on framework selection forward to the submodel
    def response_add(self, request, obj, post_url_continue=None):
        conf_dict = {
            'training_name':obj.training_name,
            'framework':obj.framework,
            'model_path':obj.model_path,
            'status':obj.status,
            'date_trained':obj.date_trained,
            'training_data_filename':obj.training_data_filename,
            'training_labels_filename':obj.training_labels_filename,
            'validation_data_filename':obj.validation_data_filename,
            'validation_labels_filename':obj.validation_labels_filename,
            'training_time':obj.training_time,
            'handle_one_hot_encoding':obj.handle_one_hot_encoding,
            'make_one_hot_encoding_task_binary':obj.make_one_hot_encoding_task_binary
    }

        if obj.framework == 'auto_sklearn':
            auto_sklearn_config = AutoSklearnConfig.objects.create(**conf_dict)
            redirect_path = '/admin/training_server/autosklearnconfig/' + str(auto_sklearn_config.id) + '/change/'

        elif obj.framework == 'auto_keras':
            auto_keras_config = AutoKerasConfig.objects.create(**conf_dict)
            redirect_path = '/admin/training_server/autokerasconfig/' + str(auto_keras_config.id) + '/change/'
        else:
            tpot_config = TpotConfig.objects.create(**conf_dict)
            redirect_path = '/admin/training_server/tpotconfig/' + str(tpot_config.id) + '/change/'

        return HttpResponseRedirect(redirect_path)

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


admin.site.register(AlgorithmConfig, AlgorithmConfigAdmin)
