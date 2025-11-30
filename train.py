import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl

from logger import Logger
from replay_buffer import ReplayBuffer
import utils

from ruamel.yaml import YAML
import pathlib
import argparse


def load_yaml(path):
    yaml = YAML()
    text = pathlib.Path(path).read_text("UTF-8")
    return yaml.load(text)


def make_env(cfg):
    suite = cfg["env"]  # "unity_env"

    if suite == "unity_env":
        from envs.unity_env import UnityEnv

        executable_path   = cfg.get("executable_path",   "path/to/build.exe")
        decision_interval = cfg.get("decision_interval", 1)
        render            = cfg.get("render", True)

        env = UnityEnv(
            executable_path=executable_path,
            decision_interval=decision_interval,
            render=render,
        )
    else:
        raise ValueError(f"unknown env: {suite}")

    return env


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        self.device = torch.device(cfg["device"])
        self.step = 0
        
        agent_name = cfg["agent"].get("name", "agent")

        self.logger = Logger(
            self.work_dir,
            save_tb=cfg["log_save_tb"],
            log_frequency=cfg["log_frequency"],
            agent= agent_name
        )

        utils.set_seed_everywhere(cfg["seed"])
        self.env = make_env(cfg)

        # compute dynamic dims
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        # 塞入 agent config
        cfg["agent"]["obs_dim"] = obs_dim
        cfg["agent"]["action_dim"] = action_dim
        cfg["agent"]["action_range"] = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max()),
        ]
        # 統一用頂層 device
        cfg["agent"]["device"] = cfg["device"]

        # instantiate agent manually
        from agent.sac import SACAgent
        self.agent = SACAgent(**cfg["agent"])

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg["replay_buffer_capacity"]),
            self.device
        )

        self.checkpoint_path = cfg.get("checkpoint_path", None)


    def evaluate(self, checkpoint_path=None):
        if checkpoint_path is not None:
            utils.load_agent(self.agent, checkpoint_path)

        avg_reward = 0
        for episode in range(self.cfg["num_eval_episodes"]):
            obs = self.env.reset()
            self.agent.reset()
            done_any = False
            ep_reward = 0.0

            while not done_any:
                with utils.eval_mode(self.agent):
                    actions = []
                    num_agents = obs.shape[0]
                    for i in range(num_agents):
                        act = self.agent.act(obs[i], sample=False) # (action_dim,)
                        actions.append(act)
                    action = np.stack(actions, axis=0).astype(np.float32)  # (num_agents, action_dim

                obs, reward, done, _ = self.env.step(action)
                ep_reward += float(np.mean(reward))
                done_any = bool(done.any())

            avg_reward += ep_reward

        avg_reward /= self.cfg["num_eval_episodes"]
        self.logger.log('eval/episode_reward', avg_reward, self.step)
        self.logger.dump(self.step)

    def run(self):
        episode = 0
        start = time.time()

        obs = self.env.reset()
        num_agents = obs.shape[0]

        episode_step = np.zeros(num_agents, dtype=np.int32)
        episode_reward = np.zeros(num_agents, dtype=np.float32)

        while self.step < self.cfg["num_train_steps"]:
            # collect action
            if self.step < self.cfg["num_seed_steps"]:
                # 隨機動作: (num_agents, action_dim)
                low = self.env.action_space.low
                high = self.env.action_space.high
                action = np.random.uniform(
                    low=low,
                    high=high,
                    size=(num_agents, self.cfg["agent"]["action_dim"])
                ).astype(np.float32)
            else:
                # SACAgent 只懂「單 obs」，所以一個 agent 一個 act
                actions = []
                with utils.eval_mode(self.agent):
                    for i in range(num_agents):
                        act = self.agent.act(obs[i], sample=True) # (action_dim,)
                        actions.append(act)
                action = np.stack(actions, axis=0).astype(np.float32)  # (num_agents, action_dim)

            # train
            if self.step >= self.cfg["num_seed_steps"]:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, done, _ = self.env.step(action)

            done = done.astype(np.float32)  # (num_agents,)
            done_no_max = done.copy()

            # ---- 多 agent -> 多筆 sample 塞進 replay buffer ----
            for i in range(num_agents):
                self.replay_buffer.add(
                    obs[i], 
                    action[i], 
                    np.array([reward[i]], dtype=np.float32), 
                    next_obs[i], 
                    float(done[i]), 
                    float(done_no_max[i]),
                    )

                episode_reward[i] += reward[i]
                episode_step[i] += 1

                if done[i] > 0.5:
                    self.logger.log(
                        'train/episode_reward', 
                        episode_reward[i], 
                        self.step
                    )
                    self.logger.log(
                        'train/episode_length', 
                        episode_step[i], 
                        self.step
                    )

                    # reset this agent
                    episode += 1
                    episode_reward[i] = 0.0
                    episode_step[i] = 0 

            obs = next_obs
            self.step += 1

            # ---- logging ----
            if self.step % self.cfg["log_frequency"] == 0:
                self.logger.log('train/episode', episode, self.step)
                self.logger.log('train/duration', time.time() - start, self.step)
                start = time.time()
                self.logger.dump(self.step, save=(self.step > self.cfg["num_seed_steps"]))

            if self.step > 0 and self.step % self.cfg["eval_frequency"] == 0:
                self.logger.log('eval/episode', episode, self.step)
                self.evaluate()

                obs = self.env.reset()
                num_agents = obs.shape[0]
                episode_step[:] = 0
                episode_reward[:] = 0.0
       
            
            # ---- checkpoint saving ----
            if self.cfg.get("Save_Agent", False):
                save_freq = self.cfg.get("save_frequency", 100000)
                if self.step % save_freq == 0:
                    ckpt_path = self.checkpoint_path
                    if ckpt_path is None:
                        ckpt_path = os.path.join(
                            self.work_dir, 
                            "checkpoints",
                            f"ckpt_{self.step}.pt"
                        )
                    utils.save_agent(self.agent, ckpt_path)

def parse_args():
    parser = argparse.ArgumentParser()

    # 指定要用哪個 config 檔
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to YAML config file."
    )

    # 指定模式：train 或 eval
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="Run mode: train or eval only."
    )

    # 可選：覆寫 seed
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override seed in config (optional)."
    )

    # 可選：指定 checkpoint path（eval 時會用到）
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to agent checkpoint (for eval mode)."
    )

    return parser.parse_args()

def main():
    args = parse_args()
    cfg = load_yaml(args.config)  

    # 2) 可選：覆寫 seed
    if args.seed is not None:
        cfg["seed"] = args.seed

    if args.checkpoint_path is not None:
        cfg["checkpoint_path"] = args.checkpoint_path

    workspace = Workspace(cfg)
    if args.mode == "train":
        workspace.run()
    elif args.mode == "eval":
        workspace.evaluate()
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == '__main__':
    main()
