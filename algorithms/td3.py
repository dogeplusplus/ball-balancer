import os
import gym
import tqdm
import yaml
import torch
import random
import inspect
import datetime
import itertools
import numpy as np
import torch.nn as nn
import torch.optim as optim

from copy import deepcopy
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.base_env import ActionTuple
from torch.utils.tensorboard import SummaryWriter
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from core import TD3Buffer, get_action
from agents import TD3ActorCritic

class TD3:
    def __init__(
        self, ac, env, epochs=100, steps_per_epoch=4000, batch_size=32,
        lr=1e-4, polyak=0.995, start_steps=10000, act_noise=0.1, target_noise=0.2, 
        noise_clip=0.5, gamma=0.99, max_ep_len=1000, update_after=1000, update_every=50, 
        policy_delay=2, model_path=None, config=None
    ):
        self.ac = ac
        self.env = env
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.polyak = polyak
        self.start_steps = start_steps
        self.act_noise = act_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.gamma = gamma
        self.policy_delay = policy_delay
        self.max_ep_len = max_ep_len
        self.update_after = update_after
        self.update_every = update_every

        self.ac_targ = deepcopy(self.ac)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
        
        if model_path is None:
            # New Model
            now = datetime.datetime.now()
            now_str = now.strftime("%Y%m%d_%H:%M:%S")
            self.model_path = f"experiments/{now_str}"
        else:
            # Existing model
            self.model_path = model_path
            self.load_model(self.model_path)

        self.writer = SummaryWriter(log_dir=self.model_path)
        self.pi_optimizer = optim.Adam(ac.pi.parameters(), lr=self.lr)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.q_optimizer = optim.Adam(self.q_params, lr=self.lr)

    def compute_loss_q(self, data):
        s, a, r, s2, d = data["state"], data["action"], data["reward"], data["next_state"], data["done"]
        s = s.float()
        a = a.float()
        r = r.float()
        s2 = s2.float()
        d = d.float()
        
        q1 = self.ac.q1(s, a)
        q2 = self.ac.q2(s, a)
        
        with torch.no_grad():
            pi_targ = self.ac_targ.pi(s2)

            # Target policy smoothing
            epsilon = torch.rand_like(pi_targ) * self.target_noise
            epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -self.ac.act_limit, self.ac.act_limit)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(s2, a2)
            q2_pi_targ = self.ac_targ.q2(s2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * q_pi_targ

        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q1 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2
        
        return loss_q

    def compute_loss_pi(self, data):
        s = data["state"]
        s = s.float()
        q1_pi = self.ac.q1(s, self.ac.pi(s))
        return -q1_pi.mean()

    def update(self, data, timer):
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        if timer % self.policy_delay == 0:

            for p in self.q_params:
                p.requires_grad = False

            self.pi_optimizer.zero_grad()
            loss_pi = self.compute_loss_pi(data)
            loss_pi.backward()
            self.pi_optimizer.step()

            for p in self.q_params:
                p.requires_grad = True

            with torch.no_grad():
                for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

            return loss_pi.detach().numpy(), loss_q.detach().numpy()

        return None, None

    def log_params(self):
        os.makedirs(self.model_path, exist_ok=True)
        attributes = inspect.getmembers(self, lambda a: not(inspect.isroutine(a)) and type(a) in (int, str, float))
        config = [a for a in attributes if not(a[0].startswith("__") and a[0].endswith("__"))]
        config = dict(config)
        config["algorithm"] = "td3"
        with open(f"{self.model_path}/config.yaml", "w") as f:
            f.write(yaml.dump(config))

    def train(self):
        # setup_pytorch_for_mpi()
        self.log_params()
        ep_len = 0
        ep_ret = 0
        s = self.env.reset()
        replay_buffer = TD3Buffer()

        episode_lengths = []
        episode_rewards = []
        pi_losses = []
        q_losses = []

        pbar = tqdm.tqdm(range(self.epochs * self.steps_per_epoch), ncols=100)
        for t in pbar:
            # Encourage exploration
            if t < self.start_steps:
                a = self.env.action_space.sample()
            else:
                a = get_action(self.ac, s, self.act_noise)

            s2, r, d, info = self.env.step(a)
            if hasattr(self.env, "render"):
                self.env.render()
            ep_len += 1
            ep_ret += r
            d = False if ep_len == self.max_ep_len else d
            
            record = dict(
                state=s,
                action=a,
                reward=np.array([r]),
                next_state=s2,
                done=np.array([d], dtype=np.float32),
            )
            replay_buffer.push(record)

            s = s2

            if d or (ep_len == self.max_ep_len):
                episode_lengths.append(ep_len)
                episode_rewards.append(ep_ret)
                o, ep_ret, ep_len = self.env.reset(), 0, 0

            if t >= self.update_after and t % self.update_every == 0:
                for j in range(self.update_every):
                    batch = replay_buffer.sample_batch(self.batch_size)
                    loss_pi, loss_q = self.update(batch, j)

                    if loss_q:
                        q_losses.append(loss_q)
                    if loss_pi:
                        pi_losses.append(loss_pi)

            if (t + 1) % self.steps_per_epoch == 0:
                epoch = (t + 1) // self.steps_per_epoch
                pbar.set_postfix(
                    dict(avg_epsiode_length=f"{np.mean(episode_lengths): .2f}")
                )
                metrics = { 
                    "Environment/Episode Length": np.mean(episode_lengths),
                    "Environment/Cumulative Reward": np.mean(episode_rewards),
                    "Losses/Policy": np.mean(pi_losses),
                    "Losses/Value": np.mean(q_losses),
                }
                episode_lengths = []
                episode_rewards = []
                pi_losses = []
                q_losses = []
                self.log_summary(epoch, metrics)

                if (epoch + 1) % 10 == 0:
                    self.save_model()

                
    def save_model(self):
        torch.save(self.ac.state_dict(), f"{self.model_path}/actor_critic")
        torch.save(self.ac_targ.state_dict(), f"{self.model_path}/actor_critic_targ")

    def log_summary(self, epoch, metrics):
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, epoch)
    
    def load_model(self, path):
        self.ac.load_state_dict(torch.load(f"{path}/actor_critic"))
        self.ac_targ.load_state_dict(torch.load(f"{path}/actor_critic_targ"))

    def test_agent(self, test_episodes):
        for j in range(test_episodes):
            s, d, ep_ret, ep_len = self.env.reset(), False, 0, 0
            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                s, r, d, _ = self.env.step(get_action(self.ac, s, 0))
                ep_ret += r
                ep_len += 1


def main():
    agent_file = "3DBall_single/3DBall_single.x86_64"
    no_graphics = True
    channel = EngineConfigurationChannel()
    unity_env = UnityEnvironment(
        file_name=agent_file, 
        seed=1, 
        no_graphics=no_graphics, 
        side_channels=[channel]
    )
    channel.set_configuration_parameters(
        time_scale=50.,
    )
    env = UnityToGymWrapper(unity_env)
    l1, l2 = 64, 64 
    activation = nn.ReLU
    output_activation = nn.Tanh
    ac = TD3ActorCritic(
        env.observation_space,
        env.action_space, 
        l1, 
        l2, 
        activation=activation
    )
    
    params = dict(
        gamma = 0.99,
        polyak = 0.995,
        act_noise = 0.1,
        target_noise = 0.2,
        epochs = 100,
        steps_per_epoch = 4000,
        start_steps = 10000,
        batch_size = 256, 
        update_after = 10000,
        update_every = 50,
        policy_delay = 2,
        lr= 1e-3,
    )

    model = TD3(ac=ac, env=env, **params)
    model.train()

if __name__ == "__main__":
    main()
    

