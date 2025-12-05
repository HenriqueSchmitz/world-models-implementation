import gymnasium as gym
import pygame
import numpy as np

env = gym.make("CarRacing-v3", render_mode="human")

print("--- Environment Created ---")
print(f"Action space: {env.action_space}")
print("  [Steering, Acceleration, Brake]")
print(f"Observation space: {env.observation_space}")
print("---------------------------")
print("\n--- Controls ---")
print("Left/Right Arrows: Steer")
print("Up Arrow: Accelerate")
print("Down Arrow: Brake")
print("R: Reset Environment")
print("Q: Quit")
print("----------------")

observation, info = env.reset()
action = np.array([0.0, 0.0, 0.0])
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
            if event.key == pygame.K_r:
                print("Resetting environment...")
                action[:] = 0.0
                observation, info = env.reset()
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
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print("Episode finished. Resetting...")
        observation, info = env.reset()
        action[:] = 0.0
print("Closing environment")
env.close()