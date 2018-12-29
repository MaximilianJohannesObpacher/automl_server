from django.contrib import admin
from django.http import HttpResponseRedirect

from training_server.models import AlgorithmConfig, AutoSklearnConfig, TpotConfig


class AlgorithmConfigAdmin(admin.ModelAdmin):

    def get_readonly_fields(self, request, obj=None):
        return ('status', 'model_path', 'date_trained', 'training_triggered', 'additional_remarks')

    def save_model(self, request, obj, form, change):
        pass # Not saving the basemodel algorithmconfig, but instead the model autosklearn_config or tpot_config in the response add method

    # depending on framework selection forward to the submodel
    def response_add(self, request, obj, post_url_continue=None):
        if obj.framework == 'auto_sklearn':
            auto_sklearn_config = AutoSklearnConfig.objects.create(
                framework=obj.framework,
                model_path=obj.model_path,
                status=obj.status,
                date_trained=obj.date_trained
            )
            redirect_path = '/admin/training_server/autosklearnconfig/' + str(auto_sklearn_config.id) + '/change/'
        else:
            tpot_config = TpotConfig.objects.create(
                framework=obj.framework,
                model_path=obj.model_path,
                status=obj.status,
                date_trained=obj.date_trained
            )
            redirect_path = '/admin/training_server/tpotconfig/' + str(tpot_config.id) + '/change/'

        return HttpResponseRedirect(redirect_path)

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

admin.site.register(AlgorithmConfig, AlgorithmConfigAdmin)
