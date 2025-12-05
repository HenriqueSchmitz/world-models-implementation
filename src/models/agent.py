from logging import Logger, getLogger
from typing import Optional

import torch
import numpy as np

from src.models.vae import ConvVAE
from src.models.worldmodel import MdnRnn
from src.models.controller import Controller

class Agent():
    def __init__(self,
                 vae_path: str,
                 rnn_path: str,
                 controller_path: str,
                 device: torch.device,
                 image_channels: int = 3,
                 vae_h_dim: int = 1024,
                 vae_z_dim: int = 32,
                 rnn_hidden_dim: int = 256,
                 num_gaussians: int = 5,
                 action_dim: int = 3,
                 reward_dim: int = 1,
                 observation_y_crop: int = 83,
                 observation_dim: int = 64,
                 logger: Optional[Logger] = None):
        self.logger = logger or getLogger(__name__)
        self.device = device
        logger.debug(f"WorldModel device: {self.device}")
        self.observation_y_crop = observation_y_crop
        self.observation_dim = observation_dim
        logger.debug(f"Creating VAE with: image_channels={image_channels} vae_h_dim={vae_h_dim} vae_z_dim={vae_z_dim}")
        self.vae = ConvVAE(image_channels, vae_h_dim, vae_z_dim).to(self.device)
        self.vae.load_state_dict(torch.load(vae_path, map_location=self.device))
        self.vae.eval()
        self.vae.requires_grad_(False)
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae = self.vae.to(self.device)
        rnn_input_dim = vae_z_dim + action_dim + reward_dim
        rnn_output_dim = vae_z_dim + reward_dim
        logger.debug(f"Creating RNN with: rnn_input_dim={rnn_input_dim} rnn_hidden_dim={rnn_hidden_dim} rnn_output_dim={rnn_output_dim} num_gaussians={num_gaussians}")
        self.worldmodel = MdnRnn(rnn_input_dim, rnn_hidden_dim, rnn_output_dim, num_gaussians)
        self.worldmodel.load_state_dict(torch.load(rnn_path, map_location=self.device))
        self.worldmodel.eval()
        self.worldmodel.requires_grad_(False)
        for param in self.worldmodel.parameters():
            param.requires_grad = False
        self.worldmodel = self.worldmodel.to(self.device)
        logger.debug(f"Creating Controller with: observation_dim={vae_z_dim} hidden_dim={rnn_hidden_dim} action_dim={action_dim}")
        self.controller = Controller(vae_z_dim, rnn_hidden_dim, action_dim)
        self.controller.load_state_dict(torch.load(controller_path, map_location=self.device))
        self.controller.eval()
        self.controller.requires_grad_(False)
        for param in self.controller.parameters():
            param.requires_grad = False
        self.controller = self.controller.to(self.device)
        self.h0 = torch.zeros(1, 1, rnn_hidden_dim).to(self.device)
        self.c0 = torch.zeros(1, 1, rnn_hidden_dim).to(self.device)
        self.hidden = (self.h0, self.c0)

    def reset(self):
        self.hidden = (self.h0, self.c0)

    def __rescale_observation_to_tensor(self, observation: np.ndarray):
        tensor = torch.from_numpy(observation).permute(2, 0, 1).float() / 255.0
        cropped_tensor = tensor[:, :self.observation_y_crop, :].to(self.device)
        resized_tensor = torch.nn.functional.interpolate(
            cropped_tensor.unsqueeze(0),
            size=(self.observation_dim, self.observation_dim),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        return resized_tensor

    def step(self, observation, reward, previous_action):
        with torch.no_grad():
            observation_tensor = self.__rescale_observation_to_tensor(observation).unsqueeze(0).to(self.device)
            reward_tensor = torch.tensor([reward], dtype=torch.float32).unsqueeze(0).to(self.device)
            action_tensor = torch.tensor(previous_action, dtype=torch.float32).unsqueeze(0).to(self.device)
            observation_representation, _, _ = self.vae.encode(observation_tensor)
            rnn_input = torch.cat([observation_representation, action_tensor, reward_tensor], dim=1)
            _, _, _, self.hidden = self.worldmodel(rnn_input.unsqueeze(0), self.hidden)
            actions = self.controller(observation_representation, self.hidden[0].squeeze(0))
            next_action = actions.cpu().numpy()[0]
        return next_action
    
    def to(self, device):
        self.device = device
        self.vae.to(device)
        self.worldmodel.to(device)
        self.controller.to(device)
        return self