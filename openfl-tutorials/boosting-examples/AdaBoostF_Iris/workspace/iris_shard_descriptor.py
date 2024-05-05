# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Mnist Shard Descriptor."""

import logging
from typing import List

import pandas as pd
from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class IrisShardDataset(ShardDataset):
    """Mnist Shard dataset class."""

    def __init__(self, x, y, data_type, rank=1, worldsize=1, complete=False):
        """Initialize TinyImageNetDataset."""
        self.data_type = data_type
        self.rank = rank
        self.worldsize = worldsize
        self.x = x if complete else x[self.rank - 1::self.worldsize]
        self.y = y if complete else y[self.rank - 1::self.worldsize]

    def get_data(self):
        return self.x, self.y

    def __getitem__(self, index: int):
        """Return an item by the index."""
        return self.x[index], self.y[index]

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.x)


class IrisShardDescriptor(ShardDescriptor):
    """Mnist Shard descriptor class."""

    def __init__(
            self,
            rank_worldsize: str = '1, 1',
            **kwargs
    ):
        """Initialize MnistShardDescriptor."""
        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))
        x_train, x_test, y_train, y_test = self.download_data()

        self.data_by_type = {
            'train': (x_train, y_train),
            'val': (x_test, y_test)
        }

    def get_shard_dataset_types(self) -> List[str]:
        """Get available shard dataset types."""
        return list(self.data_by_type)

    def get_dataset(self, dataset_type='train', complete=False):
        """Return a shard dataset by type."""
        if dataset_type not in self.data_by_type:
            raise Exception(f'Wrong dataset type: {dataset_type}')
        return IrisShardDataset(
            *self.data_by_type[dataset_type],
            data_type=dataset_type,
            rank=self.rank,
            worldsize=self.worldsize,
            complete=complete
        )

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        return ['4']

    @property
    def target_shape(self):
        """Return the target shape info."""
        return ['1']

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return (f'Iris dataset, shard number {self.rank}'
                f' out of {self.worldsize}')

    def download_data(self):
        iris = pd.read_csv("../iris_dataset/iris_data")

        X, y = iris.iloc[:, :-1].to_numpy(), iris.iloc[:, -1].to_numpy()

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        return x_train, x_test, y_train, y_test
