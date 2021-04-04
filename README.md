# ball-balancer
- Demonstration using PPO to teach a single agent to balance a ball on its head. 
- Example of training Unity ML agents using the lower level Python API, using PPO implementation from OpenAI's spinningup implementation which uses MPI for parrallel  training.
- Added support for logging in tensorboard, saving experiments and configurations for experiments.
- Also contains some legacy code for DDPG and TD3 (WIP).
- The environment is custom built from the examples given by Unity, to include only 1 agent instead of 12 due to limitations of the python wrappers.

![Ball balancing demonstration](https://github.com/dogeplusplus/ball-balancer/blob/main/assets/demo.gif)
