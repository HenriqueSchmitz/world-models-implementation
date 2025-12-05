import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Optional

import wandb
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.training.trainer import Trainer
from src.training.early_stopping import EarlyStopper


class ConvVaeTrainer(Trainer):
    def __init__(self,
                 model: torch.nn.Module,
                 weights_folder: str,
                 train_dataloader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 num_epochs: int,
                 batch_size: int,
                 train_image_log_interval: int,
                 load_checkpoint: bool = False,
                 max_norm: Optional[float] = 0.1,
                 device: Optional[torch.device] = "cpu",
                 test_dataloader: Optional[DataLoader] = None,
                 early_stopper: Optional[EarlyStopper] = None,
                 wandb_setup: Optional[dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        self.train_image_log_interval = train_image_log_interval
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

    def plot_image_comparison(self, fig, axes, input_image: Tensor, recon_image: Tensor):
        img_in = input_image.permute(1, 2, 0).cpu().numpy()
        img_in = np.clip(img_in, 0, 1)
        axes[0].imshow(img_in)
        axes[0].set_title("Original")
        axes[0].axis('off')
        img_out = recon_image.permute(1, 2, 0).cpu().numpy()
        img_out = np.clip(img_out, 0, 1)
        axes[1].imshow(img_out)
        axes[1].set_title('Reconstructed')
        axes[1].axis('off')

    def post_training_loss_logging(self, epoch: int, batch_number: int, total_training_batches: int, loss: float) -> None:
        """
        Custom logs after getting training loss for batch.
        """
        if total_training_batches % self.train_image_log_interval == 0 or total_training_batches == 1:
            input_image = self.input_images[0]
            recon_image = self.recon_batch[0]
            with torch.no_grad():
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                self.plot_image_comparison(fig, axes, input_image, recon_image)
                self.wandb_logger.log({"train/image": wandb.Image(fig)})
                plt.close(fig)

    def post_testing_loss_logging(self, epoch: int, batch_number: int, total_training_batches: int, loss: float) -> None:
        """
        Custom logs after getting testing loss for batch.
        """
        if batch_number == 1:
            input_image = self.input_images[0]
            recon_image = self.recon_batch[0]
            with torch.no_grad():
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                self.plot_image_comparison(fig, axes, input_image, recon_image)
                self.wandb_logger.log({"test/image": wandb.Image(fig)})
                plt.close(fig)

    def get_training_loss(self, data: Tensor) -> Tensor:
        """
        Returns the training loss for the given data.
        """
        self.input_images = data
        self.recon_batch, mu, logvar = self.model.forward(data)
        loss, _, _ = self.loss_fn(self.recon_batch, data, mu, logvar)
        return loss

    def get_test_loss(self, data: Tensor) -> Tensor:
        """
        Returns the training loss for the given data.
        """
        self.input_images = data
        self.recon_batch, mu, logvar = self.model.forward(data)
        loss, _, _ = self.loss_fn(self.recon_batch, data, mu, logvar)
        return loss
    
    def loss_fn(self, recon_x, x, mu, logvar):
        BCE = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD, BCE, KLD
