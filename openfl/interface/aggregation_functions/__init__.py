# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Aggregation functions package."""

from .adagrad_adaptive_aggregation import AdagradAdaptiveAggregation
from .adam_adaptive_aggregation import AdamAdaptiveAggregation
from .core import AggregationFunction
from .fedcurv_weighted_average import FedCurvWeightedAverage
from .geometric_median import GeometricMedian
from .median import Median
from .weighted_average import WeightedAverage
from .yogi_adaptive_aggregation import YogiAdaptiveAggregation
from .aggregate_random_forest import AggregateRandomForest
from .aggregate_adaboost import AggregateAdaboost

__all__ = ['Median',
           'WeightedAverage',
           'GeometricMedian',
           'AdagradAdaptiveAggregation',
           'AdamAdaptiveAggregation',
           'YogiAdaptiveAggregation',
           'AggregationFunction',
           'FedCurvWeightedAverage',
           'AggregateRandomForest',
           'AggregateAdaboost']
