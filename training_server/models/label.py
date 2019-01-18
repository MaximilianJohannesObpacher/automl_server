from django.db import models

from training_server.models import FileReformater


class Column(models.Model):
	CATEGORY = 'category' # Finite List of text_values
	NUMERIC = 'object' # string/text
	INT = 'int' # Integers
	FLOAT = 'float' # Floating_point_numbers
	BOOLEAN = 'bool' # Boolean
	DATETIME64 = 'datetime64' # Date and time values
	TIMEDELTA = 'timedelta[ns]'

	type_choices = (
		(CATEGORY, 'category'),  # Finite List of text_values
		(NUMERIC, 'object' ), # string/text
		(INT, 'int'),  # Integers
		(FLOAT, 'float'),  # Floating_point_numbers
		(BOOLEAN, 'bool'),  # Boolean
		(DATETIME64, 'datetime64'),  # Date and time values
		(TIMEDELTA, 'timedelta[ns]')
	)

	name = models.CharField(max_length=256, help_text='Name of the column')
	data_type = models.CharField(choices=type_choices, max_length=128, help_text='define the datatype this should be encoded as')
	is_label = models.BooleanField(default=False, help_text='is this a label?')
	file_reformater = models.ForeignKey(FileReformater)
