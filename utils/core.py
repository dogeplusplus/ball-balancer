import torch
import numpy as np
import yaml


def get_action(ac, state, noise_scale):
    prediction = ac.act(torch.tensor(state))
    prediction += noise_scale * np.random.randn(ac.act_dim)
    action = np.clip(prediction, -ac.act_dim, ac.act_dim)
    return action


class ReplayBuffer:
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

