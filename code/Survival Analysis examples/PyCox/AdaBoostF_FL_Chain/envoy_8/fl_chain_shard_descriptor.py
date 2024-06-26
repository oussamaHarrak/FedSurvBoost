    # Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Mnist Shard Descriptor."""

import pandas as pd
import numpy as np 
import logging
from typing import List
import random

from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor
from openfl.utilities.data_splitters.numpy import QuantitySkewLabelsSplitter
from sklearn.model_selection import train_test_split

from pycox.datasets import flchain
from sklearn.compose import make_column_selector
from sklearndf.pipeline import PipelineDF
from sklearndf.transformation import OneHotEncoderDF, ColumnTransformerDF, SimpleImputerDF, StandardScalerDF
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
logger = logging.getLogger(__name__)

random.seed(42)

class fl_chain_ShardDataset(ShardDataset):
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


class fl_chain_ShardDescriptor(ShardDescriptor):
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
        return fl_chain_ShardDataset(
            *self.data_by_type[dataset_type],
            data_type=dataset_type,
            rank=self.rank,
            worldsize=self.worldsize,
            complete=complete
        )

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        return ['8']

    @property
    def target_shape(self):
        """Return the target shape info."""
        return ['2']

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return (f'fl_chain dataset, shard number {self.rank}'
                    f' out of {self.worldsize}')
    def _split_dataframe(self , df: pd.DataFrame, split_size=0.2) :
        return train_test_split(df, stratify=df['death'], test_size=split_size)
    def download_data(self):
        df_train = flchain.read_df()
        df_test = df_train.sample(frac=0.2)
        df_train = df_train.drop(df_test.index)
        cols_standardize = ['age', 'kappa', 'lambda', 'creatinine']
        cols_leave = ['sex', 'flc.grp' , 'mgus', 'sample.yr']

        standardize = [([col], StandardScaler()) for col in cols_standardize]
        leave = [(col, None) for col in cols_leave]

        x_mapper = DataFrameMapper(standardize + leave)
        x_train = x_mapper.fit_transform(df_train).astype('float32')
        x_test = x_mapper.transform(df_test).astype('float32')
        get_target = lambda df: np.array([df['futime'].values, df['death'].values])
        y_train = get_target(df_train)
        y_test = get_target(df_test)
        y_train = pd.DataFrame(data=y_train.T, columns=['futime', 'death'])
        y_test = pd.DataFrame(data=y_test.T, columns=['futime', 'death'])
        return x_train , x_test , y_train , y_test












