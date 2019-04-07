from preprocessing.views.audio_preprocessor import AudioPreprocessorViewSet
from preprocessing.views.picture_preprocessor import PicturePreprocessorViewSet


def register_api(router):
	router.register(r'picture-preprocessor', PicturePreprocessorViewSet, base_name='picture-preprocessor')
	router.register(r'audio-preprocessor', AudioPreprocessorViewSet, base_name='audio-preprocessor')

urlpatterns = []
