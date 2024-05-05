# MAFL: Model-Agnostic Federated Learning

MAFL: Model-Agnostic Federated Learning (formerly OpenFederatedLearning-extended (OpenFL-x)) is an **open-source extension** of [Intel® OpenFL](https://github.com/securefederatedai/openfl) 1.4 supporting *federated bagging and boosting of any ML model*. The software is entirely Python-based and comes with extensive examples, as described below, exploiting [SciKit-Learn](https://scikit-learn.org/stable/) models. It has been successfully tested on x86_64, ARM and RISC-V platforms.



## Installation

It is highly recommended to create a **virtual environment** prior to installing OpenFL-x (with both [venv](https://docs.python.org/3/library/venv.html) or [conda](https://docs.conda.io/en/latest/)) to avoid compatibility issues with existing Python software already installed on the system.
Furthermore, since OpenFL-x is an extended version of OpenFL, it integrates all the features of the base framework plus the federated bagging and boosting functionalities: it is then suggested to **not install both software in the same virtual environment** since this can lead to issues in the working of both software.

You can get the last version of OpenFL-x from `pypi`:
```
pip install openfl-x
```
or, alternatively, you can clone this repository and run `pip install`:
```
git clone https://github.com/alpha-unito/OpenFL-extended.git
cd OpenFL-extended
pip install .
```
If this procedure completes successfully, you now have access to all the base features of OpenFL, plus the distributed bagging and boosting functionalities. Enjoy!



## Getting Started

The quickest way to test OpenFL-x is to execute the examples available in the `openfl-tutorials/boosting-examples` folder. There are many of them available to be run out of the box, each employing a different dataset and a different number of participants in the federation:

| Name | Algorithm | Dataset | # Train samples | # Test samples | # labels | # Features | # Envoy | ML model |
| :----------------------- | :----------------- | :----------- | --------------: | -------------: | -------: | ---------: | ------: | :----------------------- |
| `AdaBoostF_Adult` | Federated boosting | Adult | 30,132 | 15,060 | 2 | 14 | 10 | `DecisionTreeClassifier` |
| `AdaBoostF_Iris` | Federated boosting | Iris | 35 | 15 | 3 | 4 | 2 | `DecisionTreeClassifier` |
| `AdaBoostF_forestcover` | Federated boosting | ForestCover | 250,000 | 245,141 | 2 | 54 | 10 | `DecisionTreeClassifier` |
| `AdaBoostF_krvskp` | Federated boosting | KrvsKp | 2,557 | 639 | 2 | 36 | 10 | `DecisionTreeClassifier` |
| `AdaBoostF_letter` | Federated boosting | Letter | 16,000 | 4,000 | 26 | 16 | 10 | `DecisionTreeClassifier` |
| `AdaBoostF_pendigits` | Federated boosting | Pendigits | 7,494 | 3,498 | 10 | 16 | 10 | `DecisionTreeClassifier` |
| `AdaBoostF_sat` | Federated boosting | Sat | 4,435 | 2,000 | 8 | 36 | 10 | `DecisionTreeClassifier` |
| `AdaBoostF_segmentation` | Federated boosting | Segmentation | 209 | 2,099 | 7 | 19 | 10 | `DecisionTreeClassifier` |
| `AdaBoostF_splice` | Federated boosting | Splice | 2,552 | 638 | 3 | 61 | 10 | `DecisionTreeClassifier` |
| `AdaBoostF_vowel` | Federated boosting | Vowel | 792 | 198 | 11 | 27 | 10 | `DecisionTreeClassifier` |
| `AdaBoostF_vehicle` | Federated boosting | Vehicle | 677 | 169 | 4 | 18 | 10 | `DecisionTreeClassifier` |
| `RandomForest_Iris` | Federated bagging | Iris | 35 | 15 | 3 | 4 | 2 | `DecisionTreeClassifier` |

The user can customise each example by changing the data distribution across the envoys, the number of envoys itself, the ML model used as weak learner, and the federation's aggregation algorithm. More information on how to run and personalise the examples are available in the `boosting-examples` folder.

Example results of the execution of these tests and their variations are freely available on [WandB](https://wandb.ai/gmittone/AdaBoost.F?workspace=user-gmittone). During the experimentation, many SciKit-Learn classifiers have been used as weak learners, such as `ExtremelyRandomizedTree`, `RidgeLinearRegression`, `MultiLayerPerceptron`, `KNearestNeighbors`, `GaussianNaiveBayes`, and `DecisionTree`.



## Aggregation algorithms

OpenFL-x offers the possibility to experiment with basic **federated bagging** and the federated boosting method called **AdaBoost.F** developed by Polato et al. [1]. This work proposes two other federated versions of AdaBoost, namely DistBoost.F and PreWeak.F. However, AdaBoost.F has been selected as it provided the best experimental performances to the other two approaches. 

Concisely, AdaBoost.F creates *iteratively* an AdaBoost model selecting the best-performing weak learner during each federated round. It goes like this:
1. The aggregator receives the dataset size N from each collaborator and sends
them an initial version of the weak hypothesis.
2. The aggregator receives the weak hypothesis hi from each collaborator and
broadcasts the entire hypothesis space to every collaborator.
3. The errors ε committed by the global weak hypothesis on the local data are
calculated by each client and sent to the aggregator.
4. The aggregator exploits the error information to select the best weak hypothesis c, adds it to the global strong hypothesis and sends the calculated
AdaBoost coefficient α to the collaborators

More details can be found in the original paper reported below.

[1] *Polato, Mirko, Roberto Esposito, and Marco Aldinucci. "Boosting the federation: Cross-silo federated learning without gradient descent." 2022 International Joint Conference on Neural Networks (IJCNN). IEEE, 2022.*



## Performance optimisations

Together with the new aggregation algorithm made available, OpenFL-x also introduces *many other little optimisations* to the original OpenFL coda, all aiming to reduce the execution time and improve the computational performance of the code:
1. we empirically optimised the buffer sizes used by gRPC to accommodate larger models and avoid resizing operations (∼1.5% execution time improvement over the base implementation);
2. we employed the `Cloudpickle` serialisation framework over other options available (like `dill` and `pickle`) (∼2.6% execution time improvement over the base implementation);
3. we modified the TensorDB to store only the essential information of the last two federation rounds (∼14.4% execution time improvement over the base implementation);
4. we have lowered the few sleeps present in the code from 10 seconds to 0.01 seconds, which according to our experimentation on cluster systems, is the lowest value still yielding improvement (∼48.2% execution time improvement over the base implementation).

These minor optimisations achieved an overall 5.46x speedup
over the base software performance. This performance improvement makes the execution of OpenFL-x and the choice of lightweight weak learners possible on a wide range of computing platforms, even low-power ones.



## Publication

This work is currently under review at the [29th International European Conference on Parallel and Distributed Computing](https://2023.euro-par.org).
The paper's citation and link will be provided as soon as they become available.

A pre-print version of this software's paper is available on [arXiv](https://arxiv.org/abs/2303.04906).



## Contacts

This software is developed and maintained by [Gianluca Mittone](https://alpha.di.unito.it/gianluca-mittone/) (gianluca.mittone@unito.it).

