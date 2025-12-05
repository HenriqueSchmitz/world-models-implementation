import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pygame

from src.models.vae import ConvVAE
from src.models.worldmodel import MdnRnn


# --- Configuration & Constants ---
DEVICE = torch.device("mps:0" if torch.backends.mps.is_available() else "cpu")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else DEVICE)
print(f"Using device: {DEVICE}")
VAE_PATH = "./weights/vae/model.pth"
RNN_PATH = "./weights/worldmodel/model.pth"

OBSERVATION_DIM = 64
OBSERVATION_REPRESENTATION_DIM = 32
INPUT_STATE_DIM = 4  # 3 Actions + 1 Reward
OUTPUT_STATE_DIM = 1 # 1 Reward
HIDDEN_DIM = 256
RNN_INPUT_DIM = OBSERVATION_REPRESENTATION_DIM + INPUT_STATE_DIM
RNN_OUTPUT_DIM = OBSERVATION_REPRESENTATION_DIM + OUTPUT_STATE_DIM
NUM_GAUSSIANS = 5

# Display Settings
NATIVE_SIZE = 64
CROP_SIZE = (96, 83)  # The crop aspect ratio you mentioned
DISPLAY_SCALE = 6     # Scale up for visibility
DISPLAY_SIZE = (CROP_SIZE[0] * DISPLAY_SCALE, CROP_SIZE[1] * DISPLAY_SCALE)


# --- Setup Models ---
print("Loading models...")
try:
    vae = ConvVAE(image_channels=3, h_dim=1024, z_dim=OBSERVATION_REPRESENTATION_DIM).to(DEVICE)
    vae.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE))
    vae.eval()
    
    rnn = MdnRnn(input_size=RNN_INPUT_DIM, hidden_size=HIDDEN_DIM, output_size=RNN_OUTPUT_DIM, num_gaussians=NUM_GAUSSIANS).to(DEVICE)
    rnn.load_state_dict(torch.load(RNN_PATH, map_location=DEVICE))
    rnn.eval()
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please ensure VAE_PATH and RNN_PATH point to your .pth files.")
    exit(1)

# --- PyGame Setup ---
pygame.init()
screen = pygame.display.set_mode(DISPLAY_SIZE)
pygame.display.set_caption("World Model Dream")
clock = pygame.time.Clock()

# --- Simulation State ---
# 1. Initialize Hidden State (Zeroed out)
h0 = torch.zeros(1, 1, HIDDEN_DIM).to(DEVICE)
c0 = torch.zeros(1, 1, HIDDEN_DIM).to(DEVICE)
hidden = (h0, c0)

# 2. Initialize Observation (Random z)
current_z = torch.randn(1, OBSERVATION_REPRESENTATION_DIM).to(DEVICE)

# 3. Initial Reward (assumed 0 for start)
current_reward = torch.zeros(1, 1).to(DEVICE)

# Action state
action = np.array([0.0, 0.0, 0.0]) # Steering, Gas, Brake

print("--- World Model Dream Control ---")
print("Left/Right Arrows: Steer")
print("Up Arrow: Accelerate")
print("Down Arrow: Brake")
print("R: Reset Dream (New Random Z)")
print("Q: Quit")
print("---------------------------------")

running = True
while running:
    # 1. Handle Input
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
            if event.key == pygame.K_r:
                print("Resetting dream...")
                current_z = torch.randn(1, OBSERVATION_REPRESENTATION_DIM).to(DEVICE)
                h0 = torch.zeros(1, 1, HIDDEN_DIM).to(DEVICE)
                c0 = torch.zeros(1, 1, HIDDEN_DIM).to(DEVICE)
                hidden = (h0, c0)
                action[:] = 0.0

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        action[0] = -1.0
    elif keys[pygame.K_RIGHT]:
        action[0] = +1.0
    else:
        action[0] = 0.0
    if keys[pygame.K_UP]:
        action[1] = 1.0
    else:
        action[1] = 0.0
    if keys[pygame.K_DOWN]:
        action[2] = 0.8
    else:
        action[2] = 0.0

    # 2. Prepare RNN Input: Cat(z, action, reward)
    # Action + Reward shape needs to be (1, 5)
    # current_z shape is (1, 32)
    
    action_tensor = torch.tensor(action, dtype=torch.float32).to(DEVICE).unsqueeze(0) # (1, 3)
    
    # Combine inputs for RNN
    # Input is [z (32), action (3), reward (1)] -> total 36
    rnn_input = torch.cat([current_z, action_tensor, current_reward], dim=1).unsqueeze(1) # (1, 1, 36)

    # 3. Run RNN Forward Pass
    with torch.no_grad():
        pi, sigma, mu, hidden = rnn(rnn_input, hidden)
        if action[0] != 0: # If steering
            print(f"Steering: {action[0]} | Pi: {pi[0,0].cpu().numpy().round(2)}")

        # --- IMPROVED SAMPLING WITH TEMPERATURE ---
        TEMPERATURE = 0.5  # Try values between 0.5 (focused) and 1.1 (random)
        
        # 4a. Adjust Pi (Mixture Weights) with Temperature
        # Log-softmax trick: log(pi) / temp -> softmax
        # This makes dominant modes MORE dominant if temp < 1.0
        pi_log = torch.log(pi) / TEMPERATURE
        pi_adjusted = torch.softmax(pi_log, dim=-1)
        
        # 4b. Sample the Gaussian Component (k)
        categorical = torch.distributions.Categorical(pi_adjusted)
        k = categorical.sample().item()

        # 4c. Select the Mu and Sigma for component k
        chosen_mu = mu[:, :, k, :] 
        chosen_sigma = sigma[:, :, k, :]

        # 4d. Sample from the Gaussian (Scale sigma with sqrt(temp))
        # This reduces noise/jitter in the dream
        noise_scaler = np.sqrt(TEMPERATURE)
        next_data = torch.normal(chosen_mu, chosen_sigma * noise_scaler)

        # 5. Split and Decode (Same as before)
        next_z = next_data[:, :, :OBSERVATION_REPRESENTATION_DIM].squeeze(1)
        next_reward = next_data[:, :, OBSERVATION_REPRESENTATION_DIM:].squeeze(1)
        print(next_reward)
        
        # 5. Decode Z using VAE to get image
        reconstructed_img = vae.decode(next_z) # (1, 3, 64, 64)
        
        # Update current state for next iteration
        current_z = next_z
        current_reward = next_reward

    # 6. Render
    # Convert pytorch tensor to numpy for pygame
    img_np = reconstructed_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1) * 255
    img_np = img_np.astype(np.uint8)
    
    # Create PyGame Surface
    # Original output is 64x64. You requested scaling to 96x83 crop aspect ratio, then larger.
    # We will rotate and flip because PyGame treats arrays differently than plt.imshow
    surface = pygame.surfarray.make_surface(np.transpose(img_np, (1, 0, 2)))
    
    # Scale to requested view size
    surface = pygame.transform.scale(surface, DISPLAY_SIZE)
    
    screen.fill((0,0,0))
    screen.blit(surface, (0, 0))
    pygame.display.flip()
    
    clock.tick(60) # Limit to 60 FPS

pygame.quit()