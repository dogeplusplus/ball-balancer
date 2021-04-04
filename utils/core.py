import yaml
import torch
import numpy as np
import scipy.signal

from utils.mpi_tools import mpi_statistics_scalar


def get_action(ac, state, noise_scale):
    prediction = ac.act(torch.tensor(state))
    prediction += noise_scale * np.random.randn(ac.act_dim)
    action = np.clip(prediction, -ac.act_dim, ac.act_dim)
    return action


class TD3Buffer:
    def __init__(self, max_size=10000):
        self.ptr, self.size, self.max_size = 0, 0, max_size
        self.buffer = [None] * self.max_size

    def push(self, x):
        for k, v in x.items():
            if type(v) == np.array:
                x[k] == torch.tensor(v)

        self.buffer[self.ptr] = x
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = [self.buffer[idx] for idx in idxs]
        batch = {
            k: torch.tensor(np.stack([d[k] for d in batch], axis=0)) for k in batch[0].keys()
        }
        return batch


def discount_cumsum(x, discount):
    """Helper function to calculate geometric progression
    of the advantage function.

    Args:
        x (np.array): vector of advantage terms
        discount (float): discount rate

    Returns:
        Calculated advantage values.
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def combined_shape(length, shape=None):
    """Calculate the shape of actions and observations accounting
    for the environments where either could be a scalar or vector

    Args:
        length (int): number of samples
        shape (int): the dimensionality of the object

    Returns:
        Tuple specifying the desired shape of the buffer variable.
    """
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class PPOBuffer:
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
        """Push environment information into the buffer.

        Args:
            obs (np.array): observation of the environment
            act (np.array): action taken by the agent
            rew (float): reward received from taking the action
            val (float): estimated value function of the critic
            logp (float): log probability of taking action from observation state

        Raises:
            AssertionError: if the pointer exceeds the max size of replay buffer.
        """
            
        assert self.ptr < self.size
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = act
        self.rewards[self.ptr] = rew
        self.values[self.ptr] = val
        self.logp[self.ptr] = logp
        self.ptr += 1
    
    def finish_path(self, last_val=0):
        """Call this at the end of the trajectory or epoch ending.
        Looks back at buffer to see where trajectory started, uses
        rewards and value estimates from the trajectory to compute
        advantages and rewards-to-go for each state, to use as targets
        for the value function.

        Args:
            last_val (float): 0 if trajectory ended otherwise the value function
        """
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
        """Get values from the buffer for training.

        Returns:
            Dictionary of environment-agent information for training.
        """
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


def yaml2namespace(yaml_path):
    """Load the yaml file and convert it into a namespace

    Args:
        yaml_path: path to the yaml file

    Returns:
        Namespace config for the model
    """
    with open(yaml_path, 'r') as f:
        model_config_dict = yaml.load(f, yaml.FullLoader)

    model_config = Bunch(model_config_dict)
    return model_config

