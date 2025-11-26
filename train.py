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
    text = pathlib.Path(path).read_text()
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

        self.step = 0

    def evaluate(self):
        avg_reward = 0
        for episode in range(self.cfg["num_eval_episodes"]):
            obs = self.env.reset()
            self.agent.reset()
            done = False
            ep_reward = 0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, _ = self.env.step(action)
                ep_reward += reward

            avg_reward += ep_reward

        avg_reward /= self.cfg["num_eval_episodes"]
        self.logger.log('eval/episode_reward', avg_reward, self.step)
        self.logger.dump(self.step)

    def run(self):
        episode, episode_reward, done = 0, 0, True
        start = time.time()

        while self.step < self.cfg["num_train_steps"]:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start, self.step)
                    start = time.time()
                    self.logger.dump(self.step, save=(self.step > self.cfg["num_seed_steps"]))

                if self.step > 0 and self.step % self.cfg["eval_frequency"] == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward, self.step)

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # collect action
            if self.step < self.cfg["num_seed_steps"]:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # train
            if self.step >= self.cfg["num_seed_steps"]:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, done, _ = self.env.step(action)

            done_f = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done_f

            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done_f, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1

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

    return parser.parse_args()

def main():
    args = parse_args()
    cfg = load_yaml(args.config)  # 建議就用專案根目錄的 train.yaml

    # 2) 可選：覆寫 seed
    if args.seed is not None:
        cfg["seed"] = args.seed

    workspace = Workspace(cfg)
    if args.mode == "train":
        workspace.run()
    elif args.mode == "eval":
        # 簡單版：只做幾次 evaluate，不訓練
        workspace.evaluate()
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == '__main__':
    main()
