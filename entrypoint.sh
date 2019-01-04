#!/usr/bin/env bash

python manage.py migrate

# Automatically creating superuser
python manage.py shell -c "from django.contrib.auth.models import User; User.objects.create_superuser('admin', 'admin@example.com', 'test1234')"

exec "$@"