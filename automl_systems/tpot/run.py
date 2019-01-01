from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits
from tpot import TPOTClassifier

from training_server.celery import app
from training_server.models import AutoSklearnConfig


@app.task
def train_tpot(config_id):
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                        train_size=0.75, test_size=0.25)

    tpot = TPOTClassifier(generations=3, population_size=3, verbosity=2,  max_time_mins=3, max_eval_time_mins=1)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
    config = AutoSklearnConfig.objects.get(id=config_id)
    config.status = 'done!'
    config.save()
    # tpot.export('tpot_mnist_pipeline.py')