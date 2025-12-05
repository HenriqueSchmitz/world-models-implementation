import glob
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from src.metrics.wandb import WandbTrainingLogger, DummyWandbLogger
from src.training.early_stopping import EarlyStopper


class Trainer(ABC):
    def __init__(self,
                 model: torch.nn.Module,
                 weights_folder: str,
                 train_dataloader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 num_epochs: int,
                 batch_size: int,
                 load_checkpoint: bool = False,
                 max_norm: Optional[float] = 0.1,
                 device: Optional[torch.device] = "cpu",
                 test_dataloader: Optional[DataLoader] = None,
                 early_stopper: Optional[EarlyStopper] = None,
                 wandb_setup: Optional[dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        self._logger = logger if logger is not None else logging.getLogger(__name__)
        self.model = model
        self.weights_folder = weights_folder
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.load_checkpoint = load_checkpoint
        self.max_norm = max_norm
        self.device = device
        self.test_dataloader = test_dataloader
        self.early_stopper = early_stopper
        self.model.to(self.device)
        self._train_batches_per_epoch = len(self.train_dataloader)
        self._test_batches_per_epoch = len(self.test_dataloader) if self.test_dataloader is not None else 0
        self.start_epoch = 1
        self.total_trained_batches = 0
        if self.load_checkpoint:
            self.start_epoch, self.total_trained_batches = self.__load_checkpoint(self.model, self.weights_folder)
        self.wandb_setup = wandb_setup
        self.wandb_logger = self.__initialize_wandb(self.wandb_setup)
        self._logger.debug(f"Trainer initialized with: {self.__dict__}")

    def __initialize_wandb(self, wandb_setup: Optional[dict[str, Any]]) -> WandbTrainingLogger:
        if wandb_setup is None:
            return DummyWandbLogger()
        api_key = self.__get_mandatory_argument(wandb_setup, "api_key")
        project = self.__get_mandatory_argument(wandb_setup, "project")
        run_name = self.__get_mandatory_argument(wandb_setup, "run_name")
        config = self.__get_mandatory_argument(wandb_setup, "config")
        resume = self.start_epoch > 1
        return WandbTrainingLogger(api_key, project, run_name, self.model, config, resume)

    def __get_mandatory_argument(self, kwargs: dict[str, Any], argument_name: str) -> Any:
        if argument_name not in kwargs:
            raise ValueError(f"Mandatory parameter {argument_name} is missing")
        argument_value = kwargs.get(argument_name)
        if argument_value is None:
            raise ValueError(f"Mandatory parameter {argument_name} is None")
        return argument_value

    def __load_checkpoint(self, model: torch.nn.Module, weights_folder: str) -> tuple[int, int]:
        checkpoints = glob.glob(os.path.join(self.weights_folder, "epoch_*.pth"))
        if not checkpoints:
            self._logger.warning("No existing checkpoints found. Starting from scratch.")
            return 1, 0
        epochs = []
        for path in checkpoints:
            match = re.search(r'epoch_(\d+).pth', path)
            if match:
                epochs.append(int(match.group(1)))
        if not epochs:
            return 1, 0
        latest_epoch = max(epochs)
        checkpoint_path = os.path.join(self.weights_folder, f"epoch_{latest_epoch}.pth")
        self._logger.info(f"Resuming training from: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        current_global_step = latest_epoch * self._train_batches_per_epoch
        return latest_epoch + 1, current_global_step

    def train(self) -> torch.nn.Module:
        self.best_epoch = -1
        self.best_epoch_loss = float('inf')
        epochs_loop = tqdm(total=self.num_epochs, desc="Epoch", unit="epoch")
        epochs_loop.n = self.start_epoch - 1
        self.train_loop = tqdm(total=len(self.train_dataloader), desc=f"Train Epoch {self.start_epoch}", unit="batch")
        if self.test_dataloader is not None:
            self.test_loop = tqdm(total=len(self.test_dataloader), desc=f"Test Epoch {self.start_epoch}", unit="batch")
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            epochs_loop.update(1)
            epochs_loop.set_description(f"Epoch {epoch}")
            self.train_loop.reset()
            if self.test_dataloader is not None:
                self.test_loop.reset()
            self.train_epoch(epoch)
            torch.save(self.model.state_dict(), os.path.join(self.weights_folder, f"epoch_{epoch}.pth"))
            if self.test_dataloader is not None:
                should_early_stop = self.test_epoch(epoch)
                if should_early_stop:
                    break
        if self.test_dataloader is not None:
            self._logger.info(f"Best epoch: {self.best_epoch}, Loss: {self.best_epoch_loss}")
            self.model.load_state_dict(torch.load(os.path.join(self.weights_folder, f"epoch_{self.best_epoch}.pth")))
        torch.save(self.model.state_dict(), os.path.join(self.weights_folder, f"model.pth"))
        return self.model
    
    def train_epoch(self, epoch: int) -> None:
        self.model.train()
        train_loss = 0
        train_batches = 0
        self.train_loop.set_description(f"Train Epoch {epoch}")
        pre_batch_load_time = time.time()
        for batch_number, data in enumerate(self.train_dataloader):
            post_batch_load_time = time.time()
            self.train_loop.update(1)
            self.total_trained_batches += 1
            self.wandb_logger.set_step(self.total_trained_batches)
            wandb_metrics = {}
            data, batch_size = self.__prepare_batch_data(data)
            self.optimizer.zero_grad()
            pre_loss_time = time.time()
            loss = self.get_training_loss(data)
            post_loss_time = time.time()
            loss.backward()
            normalized_loss = loss.item() / batch_size
            wandb_metrics["train/loss"] = normalized_loss
            if self.max_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                wandb_metrics["train/grad_norm"] = grad_norm.item()
            self.optimizer.step()
            post_step_time = time.time()
            with torch.no_grad():
                train_loss += normalized_loss
                train_batches += 1
                self.train_loop.set_description(f"Train Epoch {epoch} Loss: {normalized_loss:.4f}")
                wandb_metrics["epoch"] = epoch
                self.wandb_logger.log(wandb_metrics)
                self._logger.debug(f"Epoch {epoch} Batch {batch_number+1} Loss: {normalized_loss:.4f}")
                self._logger.debug(f"Batch load time: {post_batch_load_time - pre_batch_load_time:.4f}s")
                self._logger.debug(f"Loss time: {post_loss_time - pre_loss_time:.4f}s")
                self._logger.debug(f"Step time: {post_step_time - post_loss_time:.4f}s")
                self.post_training_loss_logging(epoch, batch_number+1, self.total_trained_batches, normalized_loss)
            pre_batch_load_time = time.time()
        self._logger.info(f"Epoch {epoch} Loss: {train_loss / train_batches:.4f}")

    def test_epoch(self, epoch: int) -> bool:
        self.model.eval()
        test_loss = 0
        test_batches = 0
        with torch.no_grad():
            self.test_loop.reset()
            self.test_loop.set_description(f"Test Epoch {epoch}")
            for batch_number, data in enumerate(self.test_dataloader):
                self.test_loop.update(1)
                data, batch_size = self.__prepare_batch_data(data)
                loss = self.get_test_loss(data)
                normalized_loss = loss.item() / batch_size
                test_loss += normalized_loss
                test_batches += 1
                self.test_loop.set_description(f"Test Epoch {epoch} Loss: {normalized_loss:.4f}")
                self._logger.debug(f"Epoch {epoch} Batch {batch_number+1} Loss: {normalized_loss:.4f}")
                self.post_testing_loss_logging(epoch, batch_number+1, self.total_trained_batches, normalized_loss)
        avg_test_loss = test_loss / test_batches
        self._logger.info(f"Epoch {epoch} Loss: {avg_test_loss:.4f}")
        self.wandb_logger.log({"test/loss": avg_test_loss})
        if avg_test_loss < self.best_epoch_loss:
            self.best_epoch = epoch
            self.best_epoch_loss = avg_test_loss
        if self.early_stopper is not None:
            if self.early_stopper(avg_test_loss):
                self._logger.info(f"Early stopping triggered at epoch {epoch}")
                return True
        return False

    def __prepare_batch_data(self,
                             data: Union[Tensor, tuple[Tensor, ...]]) -> tuple[Union[Tensor, tuple[Tensor, ...]], int]:
        batch_size = self.batch_size
        if isinstance(data, Tensor):
            data = data.to(self.device)
            batch_size = data.size(0)
        elif isinstance(data, tuple):
            data = tuple(d.to(self.device) for d in data)
            batch_size = data[0].size(0)
        else:
            self._logger.error(f"Unsupported data type received from train_dataloader: {type(data)}")
            raise ValueError(f"Unsupported data type received from train_dataloader: {type(data)}")
        return data, batch_size

    def post_training_loss_logging(self, epoch: int, batch_number: int, total_training_batches: int, loss: float) -> None:
        """
        Custom logs after getting training loss for batch.
        """
        pass

    def post_testing_loss_logging(self, epoch: int, batch_number: int, total_training_batches: int, loss: float) -> None:
        """
        Custom logs after getting testing loss for batch.
        """
        pass

    @abstractmethod
    def get_training_loss(self, data: Union[Tensor, tuple[Tensor, ...]]) -> Tensor:
        """
        Returns the training loss for the given data.
        """
        self._logger.error("Method get_training_loss was called with no implementation")
        raise NotImplementedError("This method should be implemented by subclasses")

    @abstractmethod
    def get_test_loss(self, data: Union[Tensor, tuple[Tensor, ...]]) -> Tensor:
        """
        Returns the training loss for the given data.
        """
        self._logger.error("Method get_test_loss was called with no implementation")
        raise NotImplementedError("This method should be implemented by subclasses")