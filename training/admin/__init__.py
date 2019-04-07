from django.contrib.auth.models import Group, User

from training.admin.auto_sklearn_training import *
from training.admin.tpot_training import *
from training.admin.auto_keras_training import *

admin.site.unregister(Group)
admin.site.unregister(User)