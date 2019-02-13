from evaluation.views.validator import ValidatorViewSet


def register_api(router):
	router.register(r'validator', ValidatorViewSet, base_name='validator')

urlpatterns = []
