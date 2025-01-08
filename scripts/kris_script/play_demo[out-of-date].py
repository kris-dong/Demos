"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args
# from omni.isaac.lab_tasks.utils.parse_cfg import load_checkpoint  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
# parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--task",
    type=str,
    default="Demo-Banana-G1-Play-v0",  # Use the correct registered task
    help="Name of the task."
)

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

from g1_demos.tasks.end2end.banana.banana_perception_w_planner import Perception

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = "/scratch/kris/sim/G1-Demos/logs/rsl_rl/g1"
    print(f"[DEBUG] Experiment name: {agent_cfg.experiment_name}")
    print(f"[DEBUG] log_root_path: {log_root_path}")
    log_root_path = os.path.abspath(log_root_path)
    print(f"[DEBUG2] log_root_path: {log_root_path}")

    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    resume_path = get_checkpoint_path(log_root_path, run_dir=".*", checkpoint=".*")
    if not resume_path:
        print("[INFO] No valid checkpoint found.")
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )
    if resume_path:
        checkpoint_loaded = load_checkpoint(ppo_runner, resume_path)
    else:
        checkpoint_loaded = False

    if not checkpoint_loaded:
        print("[INFO] Starting training or simulation without a checkpoint.")

    depth_model = "vits"
    depth_input_size = 518
    yolo_model = "yolo11l"
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # create the model
    perception_model = Perception(depth_model=depth_model, depth_input_size=depth_input_size, yolo_model=yolo_model, device=device, visualize=True)


    # cv2.namedWindow("Isaac RGB", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Isaac Depth", cv2.WINDOW_NORMAL)

    camera = env.unwrapped.scene.sensors["camera"]

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        # with torch.inference_mode():
        # agent stepping
        actions = policy(obs)
        # env stepping
        obs, _, _, _ = env.step(actions)
        
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        

        # retrieve images from camera
        rgb_img = camera.data.output["rgb"]
        depth_img = camera.data.output["depth"]
        # convert from torch tensor to numpy array
        rgb_img = rgb_img[0].cpu().numpy()
        depth_img = depth_img[0].cpu().numpy()

        goal = perception_model.forward(rgb_img)

        observation_command_index = 9
        obs[:, observation_command_index:observation_command_index+3] = goal


        # Visualize Isaac Images
        depth_img = np.clip(depth_img, 0, 1e4)
        depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
        depth_img = (depth_img * 255).astype(np.uint8)

        # post process image
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        depth_img = cv2.cvtColor(depth_img, cv2.COLOR_GRAY2BGR)
        
        # cv2.imshow("Isaac RGB", rgb_img)
        cv2.imshow("Isaac Depth", depth_img)
        cv2.waitKey(1)



    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()