import numpy as np

from openfl.interface.aggregation_functions import AggregationFunction


class AggregateAdaboost(AggregationFunction):
    """Function for Adaboost.F aggregation"""

    def __init__(self, n_classes):
        super(AggregateAdaboost, self).__init__()
        self.n_classes = n_classes

    def call(self, local_tensors, *_):
        tensors = [x.tensor for x in local_tensors]

        errors = np.array([tensor[:-1] for tensor in tensors])
        norm = sum([tensor[-1] for tensor in tensors])
        wl_errs = errors.sum(axis=0) / norm

        best_model = wl_errs.argmin()
        epsilon = wl_errs.min()
        alpha = np.log((1 - epsilon) / (epsilon + 1e-10)) + np.log(self.n_classes - 1)

        return np.array([alpha, best_model])
