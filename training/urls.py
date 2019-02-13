from training.views.auto_keras_training import AutoKerasTrainingViewSet
from training.views.auto_sklearn_training import AutoSklearnTrainingViewSet
from training.views.tpot_training import TpotTrainingViewSet


def register_api(router):
	router.register(r'auto-keras-training', AutoKerasTrainingViewSet, base_name='auto-keras-config')
	router.register(r'auto-sklearn-training', AutoSklearnTrainingViewSet, base_name='auto-sklearn-config')
	router.register(r'tpot-training', TpotTrainingViewSet, base_name='tpot-config')

urlpatterns = []
