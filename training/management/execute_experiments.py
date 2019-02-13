from django.core.management import BaseCommand

from experiments_creator import start_experiment


class Command(BaseCommand):
	help = 'Starts an expirement run!'

	def handle(self, *args, **options):
		print('about to start experiment!')
		start_experiment()
		print('finished experiment!')
