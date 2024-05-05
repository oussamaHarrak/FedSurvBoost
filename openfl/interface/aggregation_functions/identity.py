from openfl.component.aggregation_functions.interface import AggregationFunction


class Identity(AggregationFunction):
    """Identity function for faking aggregation"""

    def call(self, local_tensors, *_):
        return local_tensors