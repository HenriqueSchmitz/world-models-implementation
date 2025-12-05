import random
from typing import Union, Optional, Any
from logging import Logger, getLogger

import torch
from torch import Tensor

from src.datasets.lazy_loaded_dataset import LazyLoadedDataset, SAMPLE_COUNT_METADATA


class SimulationStepsDataset(LazyLoadedDataset):
    def __init__(self,
                 data_folder: str,
                 local_data_folder: Optional[str] = None,
                 num_preloaded_files: int = 1,
                 num_workers: int = 1,
                 file_paths: Union[list[str], None] = None,
                 transform = None,
                 shuffle_files: bool = False,
                 shuffle_file_samples: bool = False,
                 random_seed: Optional[int] = None,
                 logger: Optional[Logger] = None,
                 kwargs: Optional[dict[str, Any]] = None):
        self._logger = logger if logger is not None else getLogger(self.__name__)
        if kwargs is None or "sequence_length" not in kwargs:
            self._logger.error("Argument sequence_length must be specified in kwargs")
            raise ValueError("Argument sequence_length must be specified in kwargs")
        self.sequence_length = kwargs["sequence_length"]
        super().__init__(data_folder=data_folder,
                         local_data_folder=local_data_folder,
                         num_preloaded_files=num_preloaded_files,
                         num_workers=num_workers,
                         file_paths=file_paths,
                         transform=transform,
                         shuffle_files=shuffle_files,
                         shuffle_file_samples=False, # Runs must stay in order on this dataset
                         random_seed=random_seed,
                         logger=self._logger,
                         kwargs=kwargs)

    def _process_file(self, file_path: str) -> Union[Tensor, tuple[Tensor, ...]]:
        data = torch.load(file_path)
        obs = data["observations"]
        actions = data["actions"]
        states = data["states"]
        obs = obs[:-1]
        actions = actions[1:]
        rewards = states[:-1, 0:1]
        input_states = torch.cat([actions, rewards], dim=1).float()
        output_states = rewards.float()
        if obs.ndim > 2:
            obs = obs.flatten(start_dim=1)
        limited_tensors = self.__limit_to_multiple_of_sequence_length((obs, input_states, output_states))
        sequenced_tensors = tuple(
            tensor.view(-1, self.sequence_length, *tensor.shape[1:])
            for tensor in limited_tensors
        )
        return sequenced_tensors

    def __limit_to_multiple_of_sequence_length(self, tensors: tuple[Tensor,...]) -> tuple[Tensor,...]:
        total_steps = tensors[0].shape[0]
        usable_steps = (total_steps // self.sequence_length) * self.sequence_length
        remainder = total_steps - usable_steps
        start_idx = 0
        end_idx = usable_steps
        if remainder > 0 and random.random() < 0.5:
            start_idx = remainder
            end_idx = total_steps
        return tuple(tensor[start_idx:end_idx] for tensor in tensors)

    def _calculate_total_samples(self):
        total_samples = 0
        for file_path in self.file_paths:
            run_steps = self.metadata[file_path][SAMPLE_COUNT_METADATA] - 1
            run_sequences = run_steps // self.sequence_length
            total_samples += run_sequences
        self._logger.debug(f"Total samples: {total_samples}")
        return total_samples