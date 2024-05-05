import argparse

import numpy as np
import wandb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

from adaboost import AdaBoostF
from openfl.interface.interactive_api.experiment import FLExperiment, TaskInterface, ModelInterface
from openfl.interface.interactive_api.federation import Federation
from pendigitsDataset import pendigitsDataset

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


@task_interface.register_fl_task(model='model', data_loader='train_loader', device='device', optimizer='optimizer',
                                 adaboost_coeff='adaboost_coeff', name='name', nn=False)
def train_adaboost(model, train_loader, device, optimizer, adaboost_coeff, name):
    X, y = train_loader
    X, y = np.array(X), np.array(y)
    adaboost_coeff = np.array(adaboost_coeff)

    weak_learner = model.get(0)
    ids = np.random.choice(X.shape[0], size=X.shape[0], replace=True, p=adaboost_coeff / adaboost_coeff.sum())
    weak_learner.fit(X[ids], y[ids])

    pred = weak_learner.predict(X)
    metric = accuracy_score(y, pred)
    if LOG_WANDB:
        wandb.log({"weak_train_accuracy": accuracy_score(y, pred),
                   "weak_train_precision": precision_score(y, pred, average="micro"),
                   "weak_train_recall": recall_score(y, pred, average="micro"),
                   "weak_train_f1": f1_score(y, pred, average="micro")},
                  commit=False)
    return {'accuracy': metric}


@task_interface.register_fl_task(model='model', data_loader='val_loader', device='device',
                                 adaboost_coeff='adaboost_coeff', name='name', nn=False)
def validate_weak_learners(model, val_loader, device, adaboost_coeff, name):
    X, y = val_loader
    X, y = np.array(X), np.array(y)
    adaboost_coeff = np.array(adaboost_coeff)

    rank = int(str(name).split('_')[1]) - 1

    error = []
    miss = []
    for idx, weak_learner in enumerate(model.get_estimators()):
        pred = weak_learner.predict(X)
        mispredictions = y != pred
        error.append(sum(adaboost_coeff[mispredictions]))
        miss.append(mispredictions)
        if idx == rank:
            if LOG_WANDB:
                wandb.log({"weak_test_accuracy": accuracy_score(y, pred),
                           "weak_test_precision": precision_score(y, pred, average="micro"),
                           "weak_test_recall": recall_score(y, pred, average="micro"),
                           "weak_test_f1": f1_score(y, pred, average="micro")},
                          commit=False)
    # TODO: piccolo trick, alla fine di ogni vettore errori viene mandata la norma dei pesi locali
    error.append(sum(adaboost_coeff))

    return {'errors': error}, {'misprediction': miss}


@task_interface.register_fl_task(model='model', data_loader='val_loader', device='device', nn=False)
def adaboost_update(model, val_loader, device):
    return {'adaboost_update': 0}


@task_interface.register_fl_task(model='model', data_loader='val_loader', device='device', name='name', nn=False)
def validate_adaboost(model, val_loader, device, name):
    X, y = val_loader
    pred = model.predict(np.array(X))
    f1 = f1_score(y, pred, average="micro")

    if LOG_WANDB:
        wandb.log({"test_accuracy": accuracy_score(y, pred),
                   "test_precision": precision_score(y, pred, average="micro"),
                   "test_recall": recall_score(y, pred, average="micro"),
                   "test_f1": f1_score(y, pred, average="micro")})

    return {'F1 score': f1}


federation = Federation(client_id=client_id, director_node_fqdn=args.server, director_port='50052', tls=False)
fl_experiment = FLExperiment(federation=federation, experiment_name="AdaboostF_pendigits",
                             serializer_plugin='openfl.plugins.interface_serializer.dill_serializer.DillSerializer',
                             load_default_plan=False, nn=False)
model_interface = ModelInterface(
    model=AdaBoostF(base_estimator=DecisionTreeClassifier(max_leaf_nodes=10), n_classes=10),
    optimizer=None,
    framework_plugin='openfl.plugins.frameworks_adapters.generic_adapter.GenericAdapter')
federated_dataset = pendigitsDataset()

fl_experiment.start(
    model_provider=model_interface,
    task_keeper=task_interface,
    data_loader=federated_dataset,
    rounds_to_train=args.rounds,
    opt_treatment='CONTINUE_GLOBAL'
)

fl_experiment.stream_metrics(tensorboard_logs=False)
fl_experiment.remove_experiment_data()
