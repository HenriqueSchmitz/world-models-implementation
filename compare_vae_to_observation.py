import gymnasium as gym
import pygame
import numpy as np
import torch
import torch.nn.functional as F
import cv2  # Required for video saving

from src.models.vae import ConvVAE


# --- Configuration ---
DEVICE = torch.device("mps:0" if torch.backends.mps.is_available() else "cpu")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else DEVICE)
print(f"Using device: {DEVICE}")

VAE_PATH = "./weights/vae/model.pth"
VIDEO_FILENAME = "vae_comparison.mp4"
OBSERVATION_REPRESENTATION_DIM = 32
VAE_INPUT_SIZE = 64 

# Display Sizes
PLAYER_VIEW_SIZE = (600, 600)   # Size of the window you play on
VIDEO_SCALE = 4                 # Scale up the 64x64 VAE frames for the video file
                                # Result video will be (64*2*4) wide by (64*4) high = 512x256

# --- Helper Functions ---

def preprocess_obs(obs):
    """Transforms Gym Observation (96, 96, 3) -> Tensor (1, 3, 64, 64)"""
    t = torch.from_numpy(obs).permute(2, 0, 1).float().to(DEVICE) / 255.0
    t = t[:, :83, :]
    t = t.unsqueeze(0)
    t = F.interpolate(t, size=(VAE_INPUT_SIZE, VAE_INPUT_SIZE), mode='bilinear', align_corners=False)
    return t

def tensor_to_numpy_img(tensor):
    """Transforms Tensor (1, 3, 64, 64) -> Numpy Image (64, 64, 3)"""
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    img = np.clip(img, 0, 1) * 255
    return img.astype(np.uint8)

# --- Main Setup ---

print("Loading VAE...")
try:
    vae = ConvVAE(image_channels=3, h_dim=1024, z_dim=OBSERVATION_REPRESENTATION_DIM).to(DEVICE)
    vae.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE))
    vae.eval()
    print("VAE loaded.")
except Exception as e:
    print(f"Error loading VAE: {e}")
    exit(1)

# Init Environment
env = gym.make("CarRacing-v3", render_mode="rgb_array")

# Init PyGame (For the Player View)
pygame.init()
pygame.display.set_caption("CarRacing - Manual Play")
screen = pygame.display.set_mode(PLAYER_VIEW_SIZE)
clock = pygame.time.Clock()

# Init Video Recorder (For the VAE Comparison)
# Video Frame Size: (Width = 64 * 2 * Scale, Height = 64 * Scale)
video_w = VAE_INPUT_SIZE * 2 * VIDEO_SCALE
video_h = VAE_INPUT_SIZE * VIDEO_SCALE
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video_writer = cv2.VideoWriter(VIDEO_FILENAME, fourcc, 30.0, (video_w, video_h))

print(f"--- Controls ---")
print("Play normally in the window.")
print(f"Recording VAE comparison to: {VIDEO_FILENAME}")
print("Left/Right: Steer | Up: Gas | Down: Brake | Q: Quit")

running = True
observation, info = env.reset()
action = np.array([0.0, 0.0, 0.0])

try:
    while running:
        # --- 1. PyGame Event Loop (Controls) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    observation, info = env.reset()
                    action[:] = 0.0

        keys = pygame.key.get_pressed()
        action[0] = -1.0 if keys[pygame.K_LEFT] else (1.0 if keys[pygame.K_RIGHT] else 0.0)
        action[1] = 1.0 if keys[pygame.K_UP] else 0.0
        action[2] = 0.8 if keys[pygame.K_DOWN] else 0.0

        # --- 2. Step Environment ---
        observation, reward, terminated, truncated, info = env.step(action)

        # --- 3. Render Player View (On Screen) ---
        # The raw observation is 96x96. We scale it up for the player window.
        # PyGame expects (W, H, C), Obs is (H, W, C).
        # We need to rotate/flip usually for PyGame surfarray, or just transpose.
        # Obs (96,96,3) -> Transpose -> (96,96,3) compatible with PyGame
        surface_img = np.transpose(observation, (1, 0, 2))
        surface = pygame.surfarray.make_surface(surface_img)
        surface = pygame.transform.scale(surface, PLAYER_VIEW_SIZE)
        
        screen.fill((0,0,0))
        screen.blit(surface, (0, 0))
        pygame.display.flip()

        # --- 4. Process VAE & Write Video (Background) ---
        with torch.no_grad():
            input_tensor = preprocess_obs(observation)
            recon_tensor, mu, logvar = vae(input_tensor)

        # Convert back to numpy images (64x64)
        img_in = tensor_to_numpy_img(input_tensor)
        img_out = tensor_to_numpy_img(recon_tensor)

        # Stitch side-by-side: Input | Output
        combined_frame = np.concatenate((img_in, img_out), axis=1) # Shape: (64, 128, 3)

        # Resize for better visibility in the video file
        # cv2.resize expects (Width, Height)
        final_video_frame = cv2.resize(combined_frame, (video_w, video_h), interpolation=cv2.INTER_NEAREST)

        # Convert RGB (Gym/PyTorch) to BGR (OpenCV)
        final_video_frame = cv2.cvtColor(final_video_frame, cv2.COLOR_RGB2BGR)

        # Write to file
        video_writer.write(final_video_frame)

        # Reset logic
        if terminated or truncated:
            observation, info = env.reset()
            action[:] = 0.0

        clock.tick(30) # Limit to 30 FPS for consistency with video writer

except KeyboardInterrupt:
    print("\nInterrupted.")

finally:
    print("Saving video...")
    video_writer.release()
    env.close()
    pygame.quit()
    print(f"Done. Video saved to {VIDEO_FILENAME}")