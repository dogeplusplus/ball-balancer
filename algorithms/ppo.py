import os
import yaml
import torch
import inspect
import datetime
import numpy as np

from tqdm import tqdm
from torch.optim import Adam
from gym_unity.envs import UnityToGymWrapper
from torch.utils.tensorboard import SummaryWriter
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from agents.ppo_agents import PPOActorCritic
from utils.core import PPOBuffer
from utils.mpi_tools import (
    mpi_fork,
    mpi_avg,
    proc_id,
    num_procs,
    proc_id,
    setup_pytorch_for_mpi,
    sync_params,
    mpi_avg_grads
)


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

        setup_pytorch_for_mpi()

        self.env = self.env_fn()
        self.actor_fn = actor_fn
        self.ac = actor_fn(self.env.observation_space, self.env.action_space)
        sync_params(self.ac)

        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=self.vf_lr)

        if model_path is None:
            # New Model
            now = datetime.datetime.now()
            now_str = now.strftime("%Y%m%d_%H:%M:%S")
            self.model_path = f"experiments/{now_str}_ppo"
        else:
            # Existing model
            self.model_path = model_path

        self.writer = SummaryWriter(log_dir=self.model_path)

    def compute_loss_pi(self, data):
        """Compute the loss of the actor policy.

        Args:
            data (dict): batchs of observations, actions, advantages and log probs

        Returns:
            Tuple of the policy loss and info dictionary for tensorboard logging.
        """
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
        """Compute the loss of the value critic.

        Args:
            data (dict): batch of agent-environment information

        Returns:
            Value loss.
        """
        obs, ret = data["obs"], data["ret"]
        return ((self.ac.v(obs) - ret)**2).mean()

    def update(self, data):
        """Run gradient descent on the actor and critic.

        Args:
            data (dict): batch of agent-environment information

        Returns:
            Policy loss, value loss, KL-divergence, entropy and clip fraction.
        """
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
        """Run training across multiple environments using MPI.
        """
        # Save parameters to YAML if the root process.
        if proc_id() == 0:
            self.log_params()

        seed = 10000 * proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)
        local_steps_per_epoch = int(self.steps_per_epoch / num_procs())

        obs_dim = self.env.observation_space.shape
        act_dim = self.env.action_space.shape
        replay = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, self.gamma, self.lam)
        pbar = tqdm(range(self.epochs), ncols=100)
    
        # Initial observation
        o, ep_ret, ep_len = self.env.reset(), 0, 0

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
            
    def log_params(self):
        """Log training parameters into the YAML file for later reference.
        """
        os.makedirs(self.model_path, exist_ok=True)
        attributes = inspect.getmembers(self, lambda a: not(inspect.isroutine(a)) and type(a) in (int, str, float))
        config = [a for a in attributes if not(a[0].startswith("__") and a[0].endswith("__"))]
        config = dict(config)
        config["algorithm"] = "ppo"
        with open(f"{self.model_path}/config.yaml", "w") as f:
            f.write(yaml.dump(config))

    def log_summary(self, epoch, metrics):
        """Log metrics onto tensorboard.
        """
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, epoch)
    
    def save_model(self):
        """Save model to the model directory.
        """
        torch.save(self.ac.state_dict(), f"{self.model_path}/actor_critic")

    def test_model(self, model_path, test_episodes):
        """Rollout the model on the environment for a fixed number of epsiodes.
        
        Args:
            model_path (str): path to the model directory
            test_episodes (int): number of episodes to run
        """
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


def train_environment(agent_file):
    """Make unity environment with sped up time and no visualization.

    Args:
        agent_file (str): path to the environment binary

    Returns:
        Gym environment.
    """
    time_scale = 20.
    no_graphics = True
    env = unity_env_fn(agent_file, time_scale, no_graphics, worker_id=proc_id())
    return env


def inference_environment(agent_file):
    """Create unity environment in real time with visualizations.

    Args:
        agent_file (str): path to the environment binary

    Returns:
        Gym environment.
    """
    time_scale=1.
    no_graphics=False
    env = unity_env_fn(agent_file, time_scale, no_graphics, worker_id=proc_id())
    return env


def unity_env_fn(agent_file, time_scale, no_graphics, worker_id):
    """Wrapper function for making unity environment with custom
    speed and graphics options.

    Args:
        agent_file (str): path to the environment binary
        time_scale (float): speed at which to run the simulation
        no_graphics (bool): whether or not to show the simulation

    Returns:
        Gym environment.
    """
    channel = EngineConfigurationChannel()
    unity_env = UnityEnvironment(
        file_name=agent_file, 
        no_graphics=no_graphics, 
        side_channels=[channel],
        worker_id=worker_id,
    )
    channel.set_configuration_parameters(
        time_scale=time_scale,
    )
    env = UnityToGymWrapper(unity_env)
    return env


def main():
    model_path="experiments/20210403_19:22:15_ppo"
    # model_path = None
    agent_file = "environments/3DBall_single/3DBall_single.x86_64"
    if model_path is None:
        cpus = 2
        mpi_fork(cpus)
        ppo = PPO(lambda: train_environment(agent_file), PPOActorCritic)
        ppo.train()
    else:
        cpus = 2
        mpi_fork(cpus)
        ppo = PPO(lambda: inference_environment(agent_file), PPOActorCritic)
        test_episodes = 10
        ppo.test_model(model_path, test_episodes)
 
if __name__ == "__main__":
    main()


