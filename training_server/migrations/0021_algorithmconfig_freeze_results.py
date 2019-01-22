# -*- coding: utf-8 -*-
# Generated by Django 1.11.15 on 2019-01-22 10:10
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('training_server', '0020_auto_20190122_1001'),
    ]

    operations = [
        migrations.AddField(
            model_name='algorithmconfig',
            name='freeze_results',
            field=models.BooleanField(default=False, help_text='Click this to avoid tempering with the results by making the training immutable after executing it.'),
        ),
    ]