from openfl.interface.aggregation_functions import AggregationFunction


class AggregateRandomForest(AggregationFunction):
    """Function for SciKit Learn Random Forest aggregation"""

    def call(self, local_tensors, *_):
        result = local_tensors[0].tensor
        for i in range(1, len(local_tensors)):
            model = local_tensors[i].tensor
            result.estimators_ += model.estimators_
        result.n_estimators = len(result.estimators_)

        return result
