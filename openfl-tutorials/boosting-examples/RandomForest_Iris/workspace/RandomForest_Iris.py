import argparse

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score

from IrisDataset import IrisDataset
from openfl.interface.interactive_api.experiment import FLExperiment, TaskInterface, ModelInterface
from openfl.interface.interactive_api.federation import Federation
from random_forest import MyRandomForestClassifier

LOG_WANDB = False

parser = argparse.ArgumentParser(description="Script")
parser.add_argument("--rounds", default=3, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--server", default='localhost', type=str, help="server address")
args = parser.parse_args()

random_state = np.random.RandomState(args.seed)

client_id = 'api'
cert_dir = '../cert'

task_interface = TaskInterface()


@task_interface.register_fl_task(model='model', data_loader='train_loader', device='device', optimizer='optimizer')
def train(model, train_loader, device, optimizer):
    X, y = train_loader
    X, y = np.array(X), np.array(y)
    model.fit(X, y)
    pred = model.predict(X)
    metric = accuracy_score(pred, y)

    return {'accuracy': metric}


@task_interface.register_fl_task(model='model', data_loader='val_loader', device='device')
def validate(model, val_loader, device):
    try:
        if model is not None:
            X, y = val_loader
            pred = model.predict(X)
            metric = accuracy_score(pred, y)
        else:
            print("Model not found")
            metric = 0
    except NotFittedError:
        print("Model is not yet fit")
        metric = 0

    return {'accuracy': metric}


federation = Federation(client_id=client_id, director_node_fqdn=args.server, director_port='50052', tls=False)
fl_experiment = FLExperiment(federation=federation, experiment_name="RandomForest_Iris",
                             serializer_plugin='openfl.plugins.interface_serializer.dill_serializer.DillSerializer',
                             load_default_plan=False, nn=False)
model_interface = ModelInterface(
    model=MyRandomForestClassifier(),
    optimizer=None,
    framework_plugin='openfl.plugins.frameworks_adapters.generic_adapter.GenericAdapter')
federated_dataset = IrisDataset()

fl_experiment.start(
    model_provider=model_interface,
    task_keeper=task_interface,
    data_loader=federated_dataset,
    rounds_to_train=args.rounds,
    opt_treatment='CONTINUE_GLOBAL'
)

fl_experiment.stream_metrics(tensorboard_logs=False)
fl_experiment.remove_experiment_data()
