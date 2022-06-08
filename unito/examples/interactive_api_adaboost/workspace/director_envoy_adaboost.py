import numpy as np
from openfl.interface.interactive_api.experiment import ModelInterface
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_is_fitted

from IrisDataset import IrisDataset
from MyModel import MyRandomForestClassifier
from unito.openfl_ext.experiment import FLExperiment, TaskInterface
from unito.openfl_ext.federation import Federation

random_state = np.random.RandomState(31415)

client_id = 'api'
cert_dir = '../cert'
director_node_fqdn = 'localhost'

task_interface = TaskInterface()


@task_interface.register_fl_task(model='model', data_loader='train_loader', device='device', optimizer='optimizer',
                                 adaboost_coeff='adaboost_coeff')
def train(model, train_loader, device, optimizer, adaboost_coeff):
    if model is not None:
        X, y = train_loader
        model.fit(X, y, np.array(adaboost_coeff) / np.linalg.norm(np.array(adaboost_coeff), ord=1))
        pred = model.predict(X)
        metric = accuracy_score(pred, y, normalize=True)
    else:
        metric = 0

    return {'accuracy': metric}


@task_interface.register_fl_task(model='model', data_loader='val_loader', device='device')
def validate_weak_learners(model, val_loader, device):
    miss = []
    try:
        if model is not None:
            check_is_fitted(model)

            X, y = val_loader

            errors = []
            miss = []
            for weak_learner in model.estimators_:
                pred = weak_learner.predict(X)
                errors.append(1 - accuracy_score(pred, y, normalize=True))
                miss.append([1 if x_pred != x_true else 0 for x_pred, x_true in zip(pred, y)])
        else:
            print("Model not found")
            errors = 0
    except NotFittedError:
        print("Model is not yet fit")
        errors = 0

    return {'errors': errors}, {'misprediction': miss}


@task_interface.register_fl_task(model='model', data_loader='val_loader', device='device')
def validate(model, val_loader, device):
    try:
        if model is not None:
            check_is_fitted(model)

            X, y = val_loader
            print("--------------------")
            print(model.n_estimators)
            print("--------------------")
            pred = model.predict(X)
            accuracy = accuracy_score(pred, y, normalize=True)
        else:
            print("Model not found")
            accuracy = 0
    except NotFittedError:
        print("Model is not yet fit")
        accuracy = 0

    return {'accuracy': accuracy}


@task_interface.register_fl_task(model='model', data_loader='val_loader', device='device')
def pull_model(model, val_loader, device):
    return {'pulling_model': 0}


@task_interface.register_fl_task(model='model', data_loader='val_loader', device='device')
def adaboost_update(model, val_loader, device):
    return {'adaboost_update': 0}


federation = Federation(client_id=client_id, director_node_fqdn=director_node_fqdn, director_port='50052', tls=False)
fl_experiment = FLExperiment(federation=federation, experiment_name="director_envoy_RF",
                             serializer_plugin='openfl.plugins.interface_serializer.dill_serializer.DillSerializer')  # Perché non lo prende dal plan?
model_interface = ModelInterface(model=MyRandomForestClassifier(), optimizer=None,
                                 framework_plugin='unito.openfl_ext.generic_adapter.GenericAdapter')
federated_dataset = IrisDataset(random_state=random_state)

fl_experiment.start(
    model_provider=model_interface,
    task_keeper=task_interface,
    data_loader=federated_dataset,
    rounds_to_train=3,
    opt_treatment='CONTINUE_GLOBAL',
    nn=False,
    default=False
)

# fl_experiment.stream_metrics(tensorboard_logs=False)
# fl_experiment.remove_experiment_data()
