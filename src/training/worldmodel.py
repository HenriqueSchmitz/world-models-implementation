import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Optional

import wandb
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.models.vae import ConvVAE
from src.training.trainer import Trainer
from src.training.early_stopping import EarlyStopper


class WorldModelTrainer(Trainer):
    def __init__(self,
                 model: torch.nn.Module,
                 weights_folder: str,
                 train_dataloader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 num_epochs: int,
                 batch_size: int,
                 vae: ConvVAE,
                 observation_dim: int,
                 seq_len: int,
                 load_checkpoint: bool = False,
                 max_norm: Optional[float] = 0.1,
                 device: Optional[torch.device] = "cpu",
                 test_dataloader: Optional[DataLoader] = None,
                 early_stopper: Optional[EarlyStopper] = None,
                 wandb_setup: Optional[dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        self.vae = vae
        self.observation_dim = observation_dim
        self.seq_len = seq_len
        super().__init__(model=model,
                         weights_folder=weights_folder,
                         train_dataloader=train_dataloader,
                         optimizer=optimizer,
                         num_epochs=num_epochs,
                         batch_size=batch_size,
                         load_checkpoint=load_checkpoint,
                         max_norm=max_norm,
                         device=device,
                         test_dataloader=test_dataloader,
                         early_stopper=early_stopper,
                         wandb_setup=wandb_setup,
                         logger=logger)

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

    def get_training_loss(self, data: Tensor) -> Tensor:
        """
        Returns the training loss for the given data.
        """
        obs_batch, input_states, output_states = data
        input_states = input_states
        output_states = output_states
        num_sequences = obs_batch.size(0)
        observations_encoded = self.__encode_observations(obs_batch)
        full_input_states = torch.cat((observations_encoded, input_states), dim=2)
        full_output_states = torch.cat((observations_encoded, output_states), dim=2)
        inputs = full_input_states[:, :-1, :]
        targets = full_output_states[:, 1:, :]
        h0 = torch.zeros(1, num_sequences, 256).to(self.device)
        c0 = torch.zeros(1, num_sequences, 256).to(self.device)
        hidden = (h0, c0)
        pi, sigma, mu, _ = self.model(inputs, hidden)
        loss = self.mdn_loss_fn(pi, sigma, mu, targets)
        return loss

    def get_test_loss(self, data: Tensor) -> Tensor:
        """
        Returns the training loss for the given data.
        """
        obs_batch, input_states, output_states = data
        input_states = input_states
        output_states = output_states
        num_sequences = obs_batch.size(0)
        observations_encoded = self.__encode_observations(obs_batch)
        full_input_states = torch.cat((observations_encoded, input_states), dim=2)
        full_output_states = torch.cat((observations_encoded, output_states), dim=2)
        inputs = full_input_states[:, :-1, :]
        targets = full_output_states[:, 1:, :]
        h0 = torch.zeros(1, num_sequences, 256).to(self.device)
        c0 = torch.zeros(1, num_sequences, 256).to(self.device)
        hidden = (h0, c0)
        pi, sigma, mu, _ = self.model(inputs, hidden)
        loss = self.mdn_loss_fn(pi, sigma, mu, targets)
        return loss
    
    def __encode_observations(self, observations_batch: Tensor) -> Tensor:
        num_sequences = observations_batch.size(0)
        flattened_obs = observations_batch.view(-1, 3, self.observation_dim, self.observation_dim)
        with torch.no_grad():
            observations_encoded, _, _ = self.vae.encode(flattened_obs)
            observations_encoded = observations_encoded.view(num_sequences, self.seq_len, -1)
        return observations_encoded
    
    def mdn_loss_fn(self, pi, sigma, mu, target):
        target = target.unsqueeze(2).expand_as(sigma)
        dist = torch.distributions.Normal(mu, sigma)
        log_probs = dist.log_prob(target)
        log_probs = torch.sum(log_probs, dim=3)
        log_weighted_probs = torch.log(pi) + log_probs
        log_prob_sum = torch.logsumexp(log_weighted_probs, dim=2)
        return -torch.mean(log_prob_sum)
