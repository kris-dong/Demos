"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras for demos
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import numpy as np
import torch
import cv2

from rsl_rl.runners import OnPolicyRunner
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.viplanner.config import (
    ViPlannerCarlaCfg,
    ViPlannerMatterportCfg,
    ViPlannerWarehouseCfg,
)
from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

# Import extensions to set up environment tasks
import g1_demos.tasks  # noqa: F401

from g1_demos.tasks.end2end.banana.banana_perception import Perception
from omni.viplanner.viplanner import VIPlannerAlgo
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.objects import VisualCuboid
from pxr import UsdGeom

def main():
    # Initialize the Perception model
    depth_model = "vits"
    depth_input_size = 518
    yolo_model = "yolo11l"
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    perception_model = Perception(depth_model=depth_model, depth_input_size=depth_input_size, yolo_model=yolo_model, device=device, visualize=False)

    # Initialize the ViPlanner model
    viplanner = VIPlannerAlgo(model_dir=args_cli.model_dir, device=device)


    env_cfg = ViPlannerWarehouseCfg(seed=1234)
    goal_pos = torch.tensor([3, -4.5, 1.0])

    # create environment
    env = ManagerBasedRLEnv(env_cfg)

    # # adjust the intrinsics of the camera
    # depth_intrinsic = torch.tensor([[430.31607, 0.0, 428.28408], [0.0, 430.31607, 244.00695], [0.0, 0.0, 1.0]])
    # env.scene.sensors["depth_camera"].set_intrinsic_matrices(matrices=depth_intrinsic.repeat(env.num_envs, 1, 1))
    # semantic_intrinsic = torch.tensor([[644.15496, 0.0, 639.53125], [0.0, 643.49212, 366.30880], [0.0, 0.0, 1.0]])
    # env.scene.sensors["semantic_camera"].set_intrinsic_matrices(matrices=semantic_intrinsic.repeat(env.num_envs, 1, 1))

    # reset the environment
    with torch.inference_mode():
        obs = env.reset()[0]

    # set goal cube
    VisualCuboid(
        prim_path="/World/goal",  # The prim path of the cube in the USD stage
        name="waypoint",  # The unique name used to retrieve the object from the scene later on
        position=goal_pos,  # Using the current stage units which is in meters by default.
        scale=torch.tensor([0.15, 0.15, 0.15]),  # most arguments accept mainly numpy arrays.
        size=1.0,
        color=torch.tensor([1, 0, 0]),  # RGB channels, going from 0-1
    )
    goal_pos = prim_utils.get_prim_at_path("/World/goal").GetAttribute("xformOp:translate")

    goals = torch.tensor(goal_pos.Get(), device=env.device).repeat(env.num_envs, 1)

    # Simulate physics
    while simulation_app.is_running():
        with torch.inference_mode():
            # If simulation is paused, then skip.
            if not env.sim.is_playing():
                env.sim.step(render=~args_cli.headless)
                continue

            obs = env.step(action=paths.view(paths.shape[0], -1))[0]

            # Get the raw frame from the environment's camera
            raw_frame = obs["planner_image"]["rgb_measurement"].cpu().numpy()
            raw_frame = np.transpose(raw_frame, (1, 2, 0))  # Convert to HWC format

            # Use the Perception model to get the depth image
            depth_frame = perception_model.get_depth_image(raw_frame)

        # apply planner
        goals = torch.tensor(goal_pos.Get(), device=env.device).repeat(env.num_envs, 1)
        if torch.any(
            torch.norm(obs["planner_transform"]["cam_position"] - goals)
            > viplanner.train_config.data_cfg[0].max_goal_distance
        ):
            print(
                f"[WARNING]: Max goal distance is {viplanner.train_config.data_cfg[0].max_goal_distance} but goal is {torch.norm(obs['planner_transform']['cam_position'] - goals)} away from camera position! Please select new goal!"
            )
            env.sim.pause()
            continue

        goal_cam_frame = viplanner.goal_transformer(
            goals, obs["planner_transform"]["cam_position"], obs["planner_transform"]["cam_orientation"]
        )
        _, paths, fear = viplanner.plan_dual(
            depth_frame, obs["planner_image"]["semantic_measurement"], goal_cam_frame
        )
        paths = viplanner.path_transformer(
            paths, obs["planner_transform"]["cam_position"], obs["planner_transform"]["cam_orientation"]
        )

        # draw path
        viplanner.debug_draw(paths, fear, goals)


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
