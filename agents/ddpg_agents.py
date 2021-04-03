import torch
import torch.nn as nn

class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit, l1, l2, activation=nn.ReLU, output_activation=nn.Tanh):
        super(MLPActor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, l1)
        self.ln1 = nn.LayerNorm(l1)
        self.fc2 = nn.Linear(l1, l2)
        self.ln2 = nn.LayerNorm(l2)
        self.mu = nn.Linear(l2, act_dim)
        
        self.activation = activation()
        self.output_activation = output_activation()
        self.act_limit = act_limit

    def forward(self, x):
        x = self.activation(self.ln1(self.fc1(x)))
        x = self.activation(self.ln2(self.fc2(x)))
        x = self.output_activation(self.mu(x))
        action = x * self.act_limit
        return action

class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, l1, l2, activation=nn.ReLU, output_activation=nn.Tanh):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, l1)
        self.ln1 = nn.LayerNorm(l1)
        self.fc2 = nn.Linear(l1, l2)
        self.ln2 = nn.LayerNorm(l2)

        self.activation = activation()
        self.output_activation = output_activation()

        self.action_value = nn.Linear(act_dim, l2)
        self.q = nn.Linear(l2, 1)


    def forward(self, obs, act):
        obs = self.activation(self.ln1(self.fc1(obs)))
        obs_value = self.ln2(self.fc2(obs))
        
        act_value = self.activation(self.action_value(act))
        obs_act_value = self.activation(torch.add(obs_value, act_value))
        obs_act_value = self.q(obs_act_value)
        
        return obs_act_value

class DDPGActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, l1, l2, activation=nn.ReLU, output_activation=nn.Tanh):
        super().__init__()
        self.obs_dim = observation_space.shape[0]
        self.act_dim = action_space.shape[0]
        self.act_limit = action_space.high[0]

        self.pi = MLPActor(self.obs_dim, self.act_dim, self.act_limit, l1, l2, activation, output_activation)
        self.q = MLPQFunction(self.obs_dim, self.act_dim, l1, l2, activation, output_activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()

class TD3ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, l1, l2, activation=nn.ReLU, output_activation=nn.Tanh):
        super().__init__()
        self.obs_dim = observation_space.shape[0]
        self.act_dim = action_space.shape[0]
        self.act_limit = action_space.high[0]

        self.pi = MLPActor(self.obs_dim, self.act_dim, self.act_limit, l1, l2, activation, output_activation)
        self.q1 = MLPQFunction(self.obs_dim, self.act_dim, l1, l2, activation, output_activation)
        self.q2 = MLPQFunction(self.obs_dim, self.act_dim, l1, l2, activation, output_activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()

