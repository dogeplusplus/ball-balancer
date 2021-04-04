import time
import torch
import datetime
import numpy as np
import scipy.signal
import torch.nn as nn

from tqdm import tqdm
from torch.optim import Adam
from gym_unity.envs import UnityToGymWrapper
from torch.utils.tensorboard import SummaryWriter
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from agents.ppo_agents import PPOActorCritic
from utils.mpi_tools import (
    mpi_statistics_scalar,
    mpi_fork,
    mpi_avg,
    proc_id,
    num_procs,
    proc_id,
    setup_pytorch_for_mpi,
    sync_params,
    mpi_avg_grads
)


def discount_cumsum(x, discount):
    """Magic"""
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size=10000, gamma=0.99, lam=0.95):
        self.observations = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.actions = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.advantages = np.zeros(size, dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.logp = np.zeros(size, dtype=np.float32)
    
        self.gamma = gamma
        self.lam = lam
        self.ptr, self.path_start_idx, self.size = 0, 0, size


    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.size
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = act
        self.rewards[self.ptr] = rew
        self.values[self.ptr] = val
        self.logp[self.ptr] = logp
        self.ptr += 1
    
    def finish_path(self, last_val=0):
        """Call this at the end of the trajectory or epoch ending."""
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rewards[path_slice], last_val)
        vals = np.append(self.values[path_slice], last_val)
        
        # GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.advantages[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # Rewards to go, the targets for the value function
        self.returns[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr


    def sample(self):
        assert self.ptr == self.size
        self.ptr, self.path_start_idx = 0, 0
        adv_mean, adv_std = mpi_statistics_scalar(self.advantages)
        self.advantages = (self.advantages - adv_mean) / adv_std
        data = dict(
            obs=self.observations,
            act=self.actions, 
            ret=self.returns, 
            adv=self.advantages,
            logp=self.logp
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}


class PPO:
    def __init__(self, env_fn, actor_fn, model_path=None, steps_per_epoch=4000, epochs=50, gamma=0.99,
        clip_ratio=0.2, pi_lr=3e-4, vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97,
        max_ep_len=1000, target_kl=0.01, save_freq=10):
        self.env_fn = env_fn
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.lam = lam
        self.max_ep_len = max_ep_len
        self.target_kl = target_kl
        self.save_freq = save_freq

        self.actor_fn = actor_fn

        if model_path is None:
            # New Model
            now = datetime.datetime.now()
            now_str = now.strftime("%Y%m%d_%H:%M:%S")
            self.model_path = f"experiments/{now_str}_ppo"
        else:
            # Existing model
            self.model_path = model_path
            self.load_model(self.model_path)



    def compute_loss_pi(self, data):
        obs, act, adv, logp_old = data["obs"], data["act"], data["adv"], data["logp"]

        pi, logp = self.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()

        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
    
        return loss_pi, pi_info

    def compute_loss_v(self, data):
        obs, ret = data["obs"], data["ret"]
        return ((self.ac.v(obs) - ret)**2).mean()

    def update(self, data):
        pi_l_old, pi_info_old = self.compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = self.compute_loss_v(data).item()


        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            kl = mpi_avg(pi_info["kl"])
            if kl > 1.5 * self.target_kl:
                print(f"Early stopping at {i} due to reaching max kl")
                break
            loss_pi.backward()
            mpi_avg_grads(self.ac.pi)
            self.pi_optimizer.step()

        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(self.ac.v)
            self.vf_optimizer.step()

        kl, ent, cf = pi_info["kl"], pi_info_old["ent"], pi_info["cf"]
        return pi_l_old, v_l_old, kl, ent, cf
    
    def train(self):
        self.writer = SummaryWriter(log_dir=self.model_path)
        setup_pytorch_for_mpi()
        self.env = self.env_fn()

        self.ac = self.actor_fn(self.env.observation_space, self.env.action_space)

        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=self.vf_lr)
        seed = 10000 * proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)

        start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0

        sync_params(self.ac)
        obs_dim = self.env.observation_space.shape
        act_dim = self.env.action_space.shape
        local_steps_per_epoch = int(self.steps_per_epoch / num_procs())

        replay = ReplayBuffer(obs_dim, act_dim, local_steps_per_epoch, self.gamma, self.lam)
        
        pbar = tqdm(range(self.epochs), ncols=100)
        for epoch in pbar:
            episode_lengths = []
            episode_rewards = []

            for t in range(local_steps_per_epoch):
                a, v, logp = self.ac.step(torch.as_tensor(o, dtype=torch.float32))

                next_o, r, d, _ = self.env.step(a)
                ep_ret += r
                ep_len += 1

                replay.store(o, a, r, v, logp)
                o = next_o

                timeout = ep_len == self.max_ep_len
                terminal = d or timeout
                epoch_ended = t==local_steps_per_epoch-1

                if terminal or epoch_ended:
                    if epoch_ended and not(terminal):
                        print(f"Warning: trajectory cut off by epoch at {ep_len} steps.", flush=True)

                    if timeout or epoch_ended:
                        _, v, _ = self.ac.step(torch.as_tensor(o, dtype=torch.float32))
                    else:
                        v = 0
                    replay.finish_path(v)

                    episode_lengths.append(ep_len)
                    episode_rewards.append(ep_ret)
                    o, ep_ret, ep_len = self.env.reset(), 0, 0


            data = replay.sample()
            pi_loss, value_loss, kl_div, entropy, clip_fraction = self.update(data)

            pbar.set_postfix(
                dict(avg_epsiode_length=f"{np.mean(episode_lengths): .2f}")
            )
            metrics = { 
                "Environment/Episode Length": np.mean(episode_lengths),
                "Environment/Cumulative Reward": np.mean(episode_rewards),
                "Loss/Policy": pi_loss,
                "Loss/Value": value_loss,
                "Metrics/KL Divergence": kl_div,
                "Metrics/Entropy": entropy,
                "Metrics/Clip Fraction": clip_fraction,
            }
            episode_lengths = []
            episode_rewards = []
            self.log_summary(epoch, metrics)
        
            if proc_id() == 0 and ((epoch % self.save_freq == 0) or (epoch == self.epochs - 1)):
                self.save_model()
            
    def log_summary(self, epoch, metrics):
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, epoch)
    
    def save_model(self):
        torch.save(self.ac.state_dict(), f"{self.model_path}/actor_critic")

    def test_model(self, model_path, test_episodes):
        self.env = self.env_fn()
        self.ac = self.actor_fn(self.env.observation_space, self.env.action_space)
        self.ac.load_state_dict(torch.load(f"{model_path}/actor_critic"))

        for j in range(test_episodes):
            o, d, ep_ret, ep_len = self.env.reset(), False, 0, 0
            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                with torch.no_grad():
                    a = self.ac.act(torch.as_tensor(o, dtype=torch.float32))
                o, r, d, _ = self.env.step(a)
                ep_ret += r
                ep_len += 1


def environment(agent_file):
    no_graphics=False
    channel = EngineConfigurationChannel()
    unity_env = UnityEnvironment(
        file_name=agent_file, 
        no_graphics=no_graphics, 
        side_channels=[channel]
    )
    channel.set_configuration_parameters(
        time_scale=1.,
    )
    env = UnityToGymWrapper(unity_env)
    return env


def main():
    # model_path=None
    model_path = "experiments/20210403_19:22:15_ppo"

    agent_file = "3DBall_single/3DBall_single.x86_64"
    if model_path is None:
        cpus = 2
        mpi_fork(cpus)
        ppo = PPO(lambda: environment(agent_file), PPOActorCritic)
        ppo.train()
    else:
        cpus = 1
        mpi_fork(cpus)
        ppo = PPO(lambda: environment(agent_file), PPOActorCritic)
        test_episodes = 100
        ppo.test_model(model_path, test_episodes)
 
if __name__ == "__main__":
    main()


