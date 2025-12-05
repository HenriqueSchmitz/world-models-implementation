import os
import logging
import numpy as np
from typing import Any, Dict, Tuple

import torch
import gymnasium as gym

def get_count_for_current_file(data_folder):
  """
  Reads the current count, increments it in the file, and returns the original count.
  """
  run_count_file_path = os.path.join(data_folder, "run_count.txt")
  with open(run_count_file_path, "r+") as count_file:
    count = count_file.read()
    next_count = str(int(count) + 1).zfill(7)
    count_file.seek(0)
    count_file.write(next_count)
    return count
  
def randomize_action_parameters():
  steering_bias = np.random.rand() * 2 - 1
  if steering_bias > 0:
    steering_bias = np.clip(steering_bias - 0.5, 0.0, 0.5)
  else:
    steering_bias = np.clip(steering_bias + 0.5, -0.5, 0.0)
  gas_range = (np.random.rand() + 0.2) / 1.2
  return {
    "steering_inertia": np.clip(np.random.rand() - 0.05, 0.0, 0.95),
    "steering_range": (np.random.rand() + 0.2) / 1.2,
    "steering_bias": steering_bias,
    "gas_inertia": np.clip(np.random.rand() - 0.1, 0.0, 0.9),
    "gas_range": gas_range,
    "minimum_gas": np.clip(np.random.rand() - 0.5, 0.0, 0.5) * gas_range,
    "brake_inertia": np.clip(np.random.rand() - 0.1, 0.0, 0.9),
    "brake_range": (np.random.rand() + 0.2) / 1.2,
    "block_overlap": np.random.rand() >= 0.5,
    "is_gas_priority": np.random.rand() >= 0.2,
    "maximum_overlap": np.random.rand() * 0.5
  }

def get_random_action(previous_action, parameters):
  previous_steering = previous_action[0]
  previous_gas = previous_action[1]
  previous_brake = previous_action[2]
  generated_steering = np.clip(parameters["steering_bias"] + (np.random.rand()*2 -1) * parameters["steering_range"], -parameters["steering_range"], parameters["steering_range"])
  new_steering = previous_steering*parameters["steering_inertia"] + generated_steering*(1-parameters["steering_inertia"])
  generated_gas = parameters["minimum_gas"] + np.random.rand() * (parameters["gas_range"] - parameters["minimum_gas"])
  new_gas = previous_gas*parameters["gas_inertia"] + generated_gas*(1-parameters["gas_inertia"])
  generated_brake = np.random.rand() * parameters["brake_range"]
  new_brake = previous_brake*parameters["brake_inertia"] + generated_brake*(1-parameters["brake_inertia"])
  if parameters["block_overlap"]:
    if parameters["is_gas_priority"]:
      if new_gas >= parameters["maximum_overlap"]:
        new_brake = 0
    else:
      if new_brake >= parameters["maximum_overlap"]:
        new_gas = 0
  return np.array([new_steering, new_gas, new_brake])

def get_mandatory_argument(kwargs: Dict[str, Any], argument_name: str, function_name: str) -> Any:
    argument_value = kwargs.get(argument_name, None)
    if argument_value is None:
       raise ValueError(f"Parameter {argument_name} must be provided to {function_name}")
    return argument_value

def run_single_exploration(kwargs: Dict[str, Any]) -> Tuple[int, str, int]:
    data_folder = get_mandatory_argument(kwargs, "data_folder", "runSingleExploration")
    y_crop_dim = get_mandatory_argument(kwargs, "y_crop_dim", "runSingleExploration")
    observation_dim = get_mandatory_argument(kwargs, "observation_dim", "runSingleExploration")
    logger = kwargs.get("logger") or logging.getLogger(__name__)
    
    device = torch.device("mps:0" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else device)
    action_parameters = randomize_action_parameters()
    logger.debug(action_parameters)
    env = gym.make("CarRacing-v3",
                  render_mode=None,
                  lap_complete_percent=0.95,
                  domain_randomize=False,
                  continuous=True,
                  max_episode_steps=-1)
    observation, _ = env.reset()
    action = np.array([0.0, 0.0, 0.0])
    negative_reward_streak = 0
    observations = []
    actions = []
    states = []
    while True:
        observation, reward, terminated, truncated, info = env.step(action)
        action = get_random_action(action, action_parameters)
        observations.append(rescale_to_tensor(observation, y_crop_dim, observation_dim, device))
        actions.append(build_action(action))
        states.append(build_state(reward, terminated, truncated))
        if reward < 0:
            negative_reward_streak += 1
        else:
            negative_reward_streak = 0
        if negative_reward_streak > 300:
            logger.debug("Aborting run: Car is stuck off-road.")
            break
        if terminated or truncated:
            break
    env.close()
    run_length = len(observations)
    packed_data = {
        "observations": torch.stack(observations),
        "actions": torch.stack(actions),
        "states": torch.stack(states)
    }
    del observations, actions, states
    count = get_count_for_current_file(data_folder)
    file_name = f"run_{count}.pt"
    output_file_path = os.path.join(data_folder, file_name)
    torch.save(packed_data, output_file_path)
    return int(count), file_name, run_length

def rescale_to_tensor(observation: np.ndarray, y_crop_dim: int, observation_dim: int, device: torch.device):
    tensor = torch.from_numpy(observation).permute(2, 0, 1).float() / 255.0
    cropped_tensor = tensor[:, :y_crop_dim, :].to(device)
    resized_tensor = torch.nn.functional.interpolate(
        cropped_tensor.unsqueeze(0),
        size=(observation_dim, observation_dim),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)
    return resized_tensor.to("cpu")

def build_action(action: np.array):
    return torch.from_numpy(action).float()

def build_state(reward: float, terminated: int, truncated: int):
    state = []
    state.append(float(reward))
    state.append(float(terminated))
    state.append(float(truncated))
    return torch.tensor(state).float()