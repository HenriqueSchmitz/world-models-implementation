import os
import json
import random
from abc import ABC, abstractmethod
from typing import Any, Union, Self, Optional
from logging import Logger, getLogger
from concurrent.futures import ThreadPoolExecutor

import torch
from torch import Tensor
from torch.utils.data import TensorDataset, IterableDataset
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split

SAMPLE_COUNT_METADATA = "length"
METADATA_FILE_NAME = "metadata.json"

class LazyLoadedDataset(IterableDataset, ABC):
    def __init__(self,
                 data_folder: str,
                 local_data_folder: Optional[str] = None,
                 num_preloaded_files: int = 0,
                 num_workers: int = 1,
                 file_paths: Union[list[str], None] = None,
                 transform=None,
                 shuffle_files: bool = False,
                 shuffle_file_samples: bool = False,
                 random_seed: Optional[int] = None,
                 logger: Optional[Logger] = None,
                 kwargs: Optional[dict[str, Any]] = None):
        self._logger = logger if logger is not None else getLogger(self.__name__)
        self._logger.debug(f"Initializing LazyLoadedDataset with {self.__dict__}")
        self.device = "cpu"
        self.data_folder = data_folder
        self.metadata = self.__load_metadata(data_folder, self._logger)
        if file_paths is None:
            file_paths = list(self.metadata.keys())
        self.file_paths = self.__remove_unsuccessful_file_names(file_paths, self.metadata, self._logger)
        self._logger.debug(f"Received {len(self.file_paths)} files")
        self.shuffle_files = shuffle_files
        self.shuffle_file_samples = shuffle_file_samples
        self.random_seed = random_seed
        self.transform = transform
        self.num_preloaded_files = num_preloaded_files
        self.num_workers = num_workers
        self.local_data_folder = local_data_folder
        if self.local_data_folder is not None:
            os.makedirs(self.local_data_folder, exist_ok=True)
            self._logger.debug(f"Created local data folder {self.local_data_folder}")
            if self.shuffle_file_samples:
                self._logger.warning("When using a local data folder, samples are only shuffled on the first epoch")
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        self.__total_samples = self._calculate_total_samples()
        self.__initial_load()

    def __initial_load(self):
        self._logger.debug("Resetting dataset")
        if self.shuffle_files:
            if self.random_seed is not None:
                self._logger.debug(f"Setting random seed to {self.random_seed}")
                random.seed(self.random_seed)
            self._logger.debug("Shuffling files")
            random.shuffle(self.file_paths)
        self.current_file_idx = -1
        self.current_data = None
        self.current_data_sample_count = 0
        self.current_step_idx = 0
        self.next_file_futures = []
        self.__load_next_file()

    @classmethod
    def train_test_split(cls,
                         data_folder: str,
                         local_data_folder: Optional[str] = None,
                         num_preloaded_files: int = 1,
                         num_workers: int = 1,
                         transform = None,
                         train_ratio: float = 0.8,
                         shuffle_files: bool = False,
                         shuffle_file_samples: bool = False,
                         random_seed: Optional[int] = None,
                         logger: Optional[Logger] = None,
                         kwargs: Optional[dict[str, Any]] = None) -> tuple[Self, Self]:
        logger = logger if logger is not None else getLogger(cls.__name__)
        inputs = {
            "data_folder": data_folder,
            "transform": transform,
            "train_ratio": train_ratio,
            "shuffle_files": shuffle_files,
            "shuffle_file_samples": shuffle_file_samples,
            "random_seed: ": random_seed,
            "kwargs": kwargs
        }
        logger.debug(f"Running train_test_split with {inputs}")
        metadata = cls.__load_metadata(data_folder, logger)
        file_names = cls.__remove_unsuccessful_file_names(list(metadata.keys()), metadata, logger)
        sample_counts = [metadata[filename][SAMPLE_COUNT_METADATA] for filename in file_names]
        if not file_names:
            logger.error("No valid and successfully loaded run files found.")
            raise ValueError("No valid and successfully loaded run files found.")
        split = train_test_split(
            file_names,
            sample_counts,
            test_size=1-train_ratio,
            random_state=random_seed,
            shuffle=shuffle_files
        )
        train_dataset = cls(data_folder=data_folder,
                            local_data_folder=local_data_folder,
                            num_preloaded_files=num_preloaded_files,
                            num_workers=num_workers,
                            file_paths=split[0],
                            transform=transform,
                            shuffle_files=shuffle_files,
                            shuffle_file_samples=shuffle_file_samples,
                            random_seed=random_seed,
                            logger=logger,
                            kwargs=kwargs)
        test_dataset = cls(data_folder=data_folder,
                           local_data_folder=local_data_folder,
                           num_preloaded_files=num_preloaded_files,
                           num_workers=num_workers,
                           file_paths=split[1],
                           transform=transform,
                           shuffle_files=shuffle_files,
                           shuffle_file_samples=shuffle_file_samples,
                           random_seed=random_seed,
                           logger=logger,
                           kwargs=kwargs)
        return train_dataset, test_dataset

    @staticmethod
    def __load_metadata(data_folder, logger):
        try:
            metadata_file_path = os.path.join(data_folder, METADATA_FILE_NAME)
            with open(metadata_file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Metadata file '{METADATA_FILE_NAME}' not found in '{data_folder}'.")
            raise FileNotFoundError(f"Metadata file '{METADATA_FILE_NAME}' not found in '{data_folder}'.")

    @staticmethod
    def __remove_unsuccessful_file_names(file_paths: list[str], metadata, logger):
        succesful_file_paths = []
        for file_path in file_paths:
            steps = metadata[file_path][SAMPLE_COUNT_METADATA]
            if steps > 0:
                succesful_file_paths.append(file_path)
            else:
                logger.warning(f"File '{file_path}' failed parsing with {steps} steps.")
        return succesful_file_paths

    def _calculate_total_samples(self):
        total_samples = sum(self.metadata[file_path][SAMPLE_COUNT_METADATA] for file_path in self.file_paths)
        self._logger.debug(f"Total samples: {total_samples}")
        return total_samples

    def __load_file(self, file_name: str) -> tuple[Optional[Union[TensorDataset, Tensor]], str]:
        self._logger.debug(f"Loading file {file_name}")
        file_path = os.path.join(self.data_folder, file_name)
        try:
            if self.local_data_folder is not None:
                local_file_path = os.path.join(self.local_data_folder, file_name)
                if os.path.exists(local_file_path):
                    self._logger.debug(f"Loading file {file_name} from local folder")
                    dataset = torch.load(local_file_path, weights_only=False)
                    return dataset, file_name
            processed_tensors = self._process_file(file_path)
            if isinstance(processed_tensors, Tensor):
                processed_tensors = (processed_tensors,)
                if self.shuffle_file_samples:
                    processed_tensors = self.__shuffle_tensors(processed_tensors)
                dataset = processed_tensors[0]
            else:
                if self.shuffle_file_samples:
                    processed_tensors = self.__shuffle_tensors(processed_tensors)
                dataset = TensorDataset(*processed_tensors)
            self._logger.debug(f"Successfully loaded file {file_name}")
            if self.local_data_folder is not None:
                self._logger.debug(f"Saving file {file_name} to local folder")
                torch.save(dataset, local_file_path)
            return dataset, file_name
        except Exception as e:
            self._logger.error(f"Error loading or parsing file {file_path}: {str(e)}")
            return None, file_name

    def __shuffle_tensors(self, tensors: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
        permutation = torch.randperm(tensors[0].shape[0])
        return tuple(tensor[permutation] for tensor in tensors)

    def __preload_next_file(self) -> bool:
        next_idx = self.current_file_idx + 1 + len(self.next_file_futures)
        if next_idx < len(self.file_paths):
            next_file = self.file_paths[next_idx]
            next_file_future = self.executor.submit(self.__load_file, next_file)
            self.next_file_futures.append(next_file_future)
            return True
        return False

    def __load_next_file(self):
        del self.current_data
        self.current_data = None
        self.current_step_idx = 0
        new_data = None
        new_data_sample_count = 0
        while new_data is None or new_data_sample_count <= 0:
            self.current_file_idx += 1
            if self.has_read_all_files():
                break
            if len(self.next_file_futures) > 0:
                self._logger.debug("Loading next file from preload")
                next_file_future = self.next_file_futures.pop(0)
                new_data, file_name = next_file_future.result()
                new_data_sample_count = 0 if new_data is None else len(new_data)
                self._logger.debug(f"Loaded file {file_name} with {new_data_sample_count} samples from preload")
            else:
                self._logger.debug("Loading next file directly.")
                new_data, file_name = self.__load_file(self.file_paths[self.current_file_idx])
                new_data_sample_count = 0 if new_data is None else len(new_data)
                self._logger.debug(f"Loaded file {file_name} with {new_data_sample_count} samples directly")
        if new_data is None or new_data_sample_count <= 0:
            self._logger.debug("No more files to load. Stopping iteration")
            raise StopIteration
        self.current_data = new_data
        self.current_data_sample_count = new_data_sample_count
        can_preload = True
        while len(self.next_file_futures) < self.num_preloaded_files and can_preload:
            can_preload = self.__preload_next_file()

    def has_read_all_files(self) -> bool:
        return self.current_file_idx >= len(self.file_paths) and len(self.next_file_futures) <= 0

    def __len__(self):
        return self.__total_samples

    def __iter__(self):
        self.__initial_load()
        return self

    def __next__(self):
        if self.__is_current_file_exhausted():
            self._logger.debug("Current file exhausted. Loading next file")
            self.__load_next_file()
        if self.current_data is None:
            self._logger.error("No data loaded. Check dataset initialization")
            raise IndexError("No data loaded. Check dataset initialization")
        sample = self.current_data[self.current_step_idx]
        self.current_step_idx += 1
        if self.transform is not None:
            return self.transform(sample)
        return sample

    def __is_current_file_exhausted(self):
        return self.current_data is not None and self.current_step_idx >= self.current_data_sample_count

    @abstractmethod
    def _process_file(self, file_path: str) -> Union[Tensor, tuple[Tensor, ...]]:
        """"
        Reads a file and returns a list of tensors.
        """
        self._logger.error("Method _process_file was called with no implementation")
        raise NotImplementedError("This method should be implemented by subclasses")