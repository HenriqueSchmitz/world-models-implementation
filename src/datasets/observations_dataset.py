from typing import Union

import torch
from torch import Tensor

from src.datasets.lazy_loaded_dataset import LazyLoadedDataset


class ObservationsDataset(LazyLoadedDataset):
    def _process_file(self, file_path: str) -> Union[Tensor, tuple[Tensor, ...]]:
        data = torch.load(file_path)
        observations = data["observations"]
        return observations