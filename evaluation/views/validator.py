from rest_framework import viewsets

from evaluation.models.validator import Validator
from evaluation.serializers.validator import ValidatorSerializer


class ValidatorViewSet(viewsets.ModelViewSet):
	queryset = Validator.objects.all()
	serializer_class = ValidatorSerializer