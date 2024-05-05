# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Assigner package."""

from .assigner import Assigner
from .random_grouped_assigner import RandomGroupedAssigner
from .static_grouped_assigner import StaticGroupedAssigner
from .adaboost_assigner import AdaBoostAssigner


__all__ = [
    'Assigner',
    'RandomGroupedAssigner',
    'StaticGroupedAssigner',
    'AdaBoostAssigner'
]
