from training_server.views.auto_keras_config import AutoKerasConfigViewSet
from training_server.views.auto_sklearn_config import AutoSklearnConfigViewSet
from training_server.views.tpot_config import TpotConfigViewSet


def register_api(router):
	router.register(r'auto-keras-config', AutoKerasConfigViewSet, base_name='auto-keras-config')
	router.register(r'auto-sklearn-config', AutoSklearnConfigViewSet, base_name='auto-sklearn-config')
	router.register(r'tpot-config', TpotConfigViewSet, base_name='tpot-config')

urlpatterns = []
