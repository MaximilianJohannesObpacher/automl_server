# -*- coding: utf-8 -*-
# Generated by Django 1.11.15 on 2019-01-19 20:17
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('training_server', '0016_filereformater_additional_remarks'),
    ]

    operations = [
        migrations.AlterField(
            model_name='audioreformater',
            name='folder_name',
            field=models.CharField(blank=True, default='', max_length=256, null=True),
        ),
    ]
