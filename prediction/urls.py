from prediction.views.predictor import PredictorViewSet

def register_api(router):
	router.register(r'predictor', PredictorViewSet, base_name='predictor')


urlpatterns = []
