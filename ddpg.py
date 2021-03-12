import gym
import tqdm
import torch
import random
import numpy as np
import datetime
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)

from agents import MLPActorCritic

# Create the StringLogChannel class
class ReplayBuffer:
    def __init__(self):
        self.buffer = []

    def push(self, x):
        for k, v in x.items():
            if type(v) == np.array:
                x[k] == torch.tensor(v)
        self.buffer.append(x)

    def sample_batch(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        batch = {
            k: torch.tensor(np.stack([d[k] for d in batch], axis=0)) for k in batch[0].keys()
        }
        return batch


def get_action(ac, state, noise_scale):
    prediction = ac.pi(torch.tensor(state)).detach().numpy()
    prediction += noise_scale * np.random.randn(ac.act_dim)
    action = np.clip(prediction, -ac.act_dim, ac.act_dim)
    return action


class DDPG:
    def __init__(self, ac, env, epochs=100, steps_per_epoch=4000, batch_size=32,
        lr=1e-3, polyak=0.995, start_steps=10000, noise_scale=0.1, gamma=0.99,
        max_ep_len=1000, update_after=1000, update_every=50, model_path=None):

        self.ac = ac
        self.env = env
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.polyak = polyak
        self.start_steps = start_steps
        self.noise_scale = noise_scale
        self.gamma = gamma
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
        self.q_optimizer = optim.Adam(ac.q.parameters(), lr=self.lr)

    def compute_loss_q(self, data):
        s, a, r, s2, d = (
            data["state"],
            data["action"],
            data["reward"],
            data["next_state"],
            data["done"],
        )

        q = self.ac.q(s, a)

        with torch.no_grad():
            q_pi_targ = self.ac_targ.q(s2, self.ac_targ.pi(s2))
            backup = r + gamma * (1 - d) * q_pi_targ

        loss_q = ((q - backup) ** 2).mean()
        return loss_q

    def compute_loss_pi(self, data):
        s = data["state"]
        q_pi = self.ac.q(s, self.ac.pi(s))
        return -q_pi.mean()

    def update(self, data):  # First run one gradient descent step for Q.
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for p in ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in ac.q.parameters():
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
        
        return loss_pi.detach().numpy(), loss_q.detach().numpy()

    def train(self):
        ep_len = 0
        ep_ret = 0
        s = self.env.reset()
        replay_buffer = ReplayBuffer()

        episode_lengths = []
        pi_losses = []
        q_losses = []

        pbar = tqdm.tqdm(range(self.epochs * self.steps_per_epoch), ncols=100)
        for t in pbar:
            # Encourage exploration
            if t < self.start_steps:
                a = self.env.action_space.sample()
            else:
                a = get_action(self.ac, s, self.noise_scale)

            s2, r, d, info = self.env.step(a)
            ep_len += 1
            ep_ret += r
            s = s2
            d = False if ep_len == self.max_ep_len else d
            
            record = dict(
                state=s,
                action=np.array([a]),
                reward=np.array([r]),
                next_state=s2,
                done=np.array([d], dtype=np.float32),
            )
            replay_buffer.push(record)

            if d or (ep_len == self.max_ep_len):
                episode_lengths.append(ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0

            if t >= self.update_after and t % self.update_every == 0:
                pbar.set_postfix(
                    dict(avg_epsiode_length=f"{np.mean(episode_lengths): .2f}")
                )
                for _ in range(update_every):
                    batch = replay_buffer.sample_batch(batch_size)
                    loss_pi, loss_q = self.update(batch)
                    q_losses.append(loss_q)
                    pi_losses.append(loss_pi)

            if (t + 1) % self.steps_per_epoch == 0:
                epoch = (t + 1) // self.steps_per_epoch
                metrics = dict(
                    eps_len=np.mean(episode_lengths),
                    loss_pi=np.mean(pi_losses),
                    loss_q=np.mean(q_losses),
                )
                episode_lengths = []
                pi_losses = []
                q_losses = []
                self.log_summary(epoch, metrics)
                
    def save(self):
        torch.save(self.ac.state_dict(), f"{self.model_path}/actor_critic")
        torch.save(self.ac_targ.state_dict(), f"{self.model_path}/actor_critic_targ")

    def log_summary(self, epoch, metrics):
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, epoch)
    
    def load_model(self, path):
        self.ac = self.ac.load_state_dict(torch.load(path))
        self.ac_targ = self.ac_targ.load_state_dict(torch.load(path))


if __name__ == "__main__":
    agent_file = "3DBall_single/3DBall_single.x86_64"
    no_graphics = False
    unity_env = UnityEnvironment(file_name=agent_file, seed=1, no_graphics=no_graphics)
    env = UnityToGymWrapper(unity_env)

    # import gym
    # env = gym.make("MountainCarContinuous-v0")
    gamma = 0.99
    polyak = 0.995
    l1, l2 = 256, 256
    noise_scale = 0.1
    activation = nn.ReLU
    output_activation = nn.Tanh
    epochs = 100
    steps_per_epoch = 40
    start_steps = 10
    batch_size = 32
    update_after = 10
    update_every = 50

    ac = MLPActorCritic(env.observation_space, env.action_space, l1, l2)
    model = DDPG(
        ac=ac,
        env=env,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        noise_scale=noise_scale,
        start_steps=start_steps,
        batch_size=batch_size,
        polyak=polyak,
        update_after=update_after,
        update_every=update_every,
    )
    model.train()
