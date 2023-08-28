from dataclasses import dataclass
from os.path import join

import pandas as pd
import numpy as np

from alpaca.system.files import DATA_DIR
from alpaca.system.frameworks.numpy_classes import NumpyArrayDataLoader
from alpaca.system.interfaces.dataset import Dataset
from alpaca.system.factory import DatasetFactory, DataState


@dataclass
class CSVLoadFun:
    parent_dir: str
    dataset_name: str
    ignore_first_column: bool

    def __call__(self) -> Dataset:
        path = join(DATA_DIR, self.parent_dir, self.dataset_name + ".csv")
        df = pd.read_csv(path)
        if not "y" in df.columns:
            raise RuntimeError(
                f'load_dataset_from_csv: Dataset has no column "y"! Loaded from: {path}'
            )
        x_raw = df.loc[:, df.columns != "y"].to_numpy(dtype=np.float32)
        if self.ignore_first_column:
            x_raw = x_raw[:, 1:]
        y_raw = df["y"].to_numpy(dtype=np.float32)
        return Dataset(NumpyArrayDataLoader(x_raw, y_raw))


_NUMPY_DATASETS = [
    ("processed", "esol_features_6dims", True),
    ("processed", "esol_mol2vec_100dims", False),
    ("processed", "esol_pca_100dims", True),
    ("processed", "lipo_mol2vec_100dims", False),
    ("processed", "lipo_pca_100dims", True),
    ("processed", "sampl_mol2vec_100dims", False),
    ("processed", "sampl_pca_100dims", True),
    ("raw", "synthetic_2dims", False),
]

for dir, name, ignore_first_column in _NUMPY_DATASETS:
    DatasetFactory.register(
        name, CSVLoadFun(dir, name, ignore_first_column)
    )

@dataclass
class DataStateGetter:
    dataset_key: str

    def __call__(self) -> DataState:
        return DataState(dataset_key=self.dataset_key)

NUMPY_DATASETS = {
    tpl[1] : DataStateGetter(tpl[1]) for tpl in _NUMPY_DATASETS
}
