from django.contrib.auth.models import Group, User

from training_server.admin.auto_sklearn_config import *
from training_server.admin.tpot_config import *
from training_server.admin.auto_keras_config import *

admin.site.unregister(Group)
admin.site.unregister(User)