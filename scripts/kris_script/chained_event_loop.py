
import argparse
import os
import torch
import numpy as np
import cv2

# Omni Isaac Sim imports
from omni.isaac.lab.app import AppLauncher
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab_tasks.utils import parse_env_cfg

# Perception and Planning imports
from g1_demos.tasks.end2end.banana.banana_perception_w_planner import Perception
from g1_demos.tasks.end2end.banana.viplanner.viplanner import VIPlannerAlgo

# Parse arguments
parser = argparse.ArgumentParser(description="Chain perception with planning in a unified event loop.")
parser.add_argument("--scene", default="warehouse", choices=["matterport", "carla", "warehouse"], type=str, help="Scene to load.")
parser.add_argument("--model_dir", default=None, type=str, help="Path to ViPlanner model directory.")
parser.add_argument("--task", type=str, default=None, help="Task name for Isaac Sim environment.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# Launch Omniverse application
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

def main():
    # Environment and planner configuration
    if args_cli.scene == "warehouse":
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device)
        goal_pos = torch.tensor([3, -4.5, 1.0])
    else:
        raise NotImplementedError(f"Scene {args_cli.scene} not yet supported!")

    # Initialize environment
    env = ManagerBasedRLEnv(env_cfg)
    env.reset()

    # Perception model setup
    perception_model = Perception(
        depth_model="vits",
        depth_input_size=518,
        yolo_model="yolo11l",
        device="cuda" if torch.cuda.is_available() else "cpu",
        visualize=True
    )

    # Planning algorithm setup
    viplanner = VIPlannerAlgo(model_dir=args_cli.model_dir, device=env.device)

    # Camera setup
    camera = env.scene.sensors["camera"]

    # Simulation loop
    while simulation_app.is_running():
        obs = env.step(action=None)[0]

        # Perception
        rgb_img = camera.data.output["rgb"][0].cpu().numpy()
        goal = perception_model.forward(rgb_img).cpu().numpy()

        # Transform perception output into planner goals
        goals = torch.tensor(goal, device=env.device).repeat(env.num_envs, 1)
        
        # Planning
        _, paths, fear = viplanner.plan_dual(
            obs["planner_image"]["depth_measurement"],
            obs["planner_image"]["semantic_measurement"],
            goals
        )

        # Simulate using the planned path
        env.step(action=paths.view(paths.shape[0], -1))

        # Visualize results if needed
        perception_model.show_debug_view(obs["planner_image"]["depth_measurement"], [], [])

    # Cleanup
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
