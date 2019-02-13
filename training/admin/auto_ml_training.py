from django.contrib import admin
from django.http import HttpResponseRedirect

from training.models import AutoMlTraining, AutoSklearnTraining, TpotTraining, AutoKerasTraining


class AutoMlTrainingAdmin(admin.ModelAdmin):

    def get_readonly_fields(self, request, obj=None):
        return ('status', 'model_path', 'date_trained', 'training_triggered', 'additional_remarks', 'training_time')

    def save_model(self, request, obj, form, change):
        pass # Not saving the basemodel AutoMlTraining, but instead the model autosklearn_config or tpot_training in the response add method


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
            'label_one_hot_encoding_binary':obj.label_one_hot_encoding_binary
    }

        if obj.framework == 'auto_sklearn':
            auto_sklearn_training = AutoSklearnTraining.objects.create(**conf_dict)
            redirect_path = '/admin/training/autosklearntraining/' + str(auto_sklearn_training.id) + '/change/'

        elif obj.framework == 'auto_keras':
            auto_keras_training = AutoKerasTraining.objects.create(**conf_dict)
            redirect_path = '/admin/training/autokerastraining/' + str(auto_keras_training.id) + '/change/'
        else:
            tpot_training = TpotTraining.objects.create(**conf_dict)
            redirect_path = '/admin/training/tpottraining/' + str(tpot_training.id) + '/change/'

        return HttpResponseRedirect(redirect_path)

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


admin.site.register(AutoMlTraining, AutoMlTrainingAdmin)
