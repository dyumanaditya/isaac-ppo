# Isaac Proximal Policy Optimization
This repository contains an implementation of the RL Algorithm Proximal Policy Optimization.
The implementation is based on the paper [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) by Schulman et al.
and is inspired from [RSL_RL](https://github.com/leggedrobotics/rsl_rl/tree/master).


It is meant to be used as alongside [Isaac Sim Orbit](https://isaac-orbit.github.io/). This allows us to train agents
using multiple parallel environments. The default hyperparameters are optimized for Isaac Sim Orbit.


## Installation
### Isaac Sim Orbit
To use this package make sure Isaac Sim and Orbit have been installed correctly. Installation instructions can be found here
[Isaac Sim](https://isaac-orbit.github.io/orbit/source/setup/installation.html#installing-isaac-sim) and [Isaac Sim Orbit](https://isaac-orbit.github.io/orbit/source/setup/installation.html#installing-orbit).
Make sure you setup the virtual environment for orbit and activate it before proceeding.
### Isaac PPO
```bash
pip install .
```
Or in development mode
```bash
pip install -e .
```


## Usage
We only support continuous action spaces at the moment.

### Training
```python
import gymnasium as gym
from ppo import PPO, Hyperparameters

from omni.isaac.orbit.app.app_launcher import AppLauncher
from utils.argparser import get_argparser

"""
Launch Isaac Sim as global variables
"""
parser = get_argparser()

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# Parse the arguments
args_cli = parser.parse_args()

# Launch Omniverse
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import the necessary modules after Isaac Sim is launched
from omni.isaac.orbit_tasks.utils import parse_env_cfg


# Parse the arguments
# Environment configuration
env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)

# Create the environment
env = gym.make(args_cli.task, cfg=env_cfg)

# Create the hyperparameters object (modify anything if needed)
hyperparameters = Hyperparameters()
hyperparameters.lr = 1e-4

# Create the agent
agent = PPO(env, hyperparameters)
agent.policy_path = 'policies'

# Learn
agent.learn(max_steps=2000)

```

### Evaluation
```python
import gymnasium as gym
from ppo import PPO, Hyperparameters

from omni.isaac.orbit.app.app_launcher import AppLauncher
from utils.argparser import get_argparser

"""
Launch Isaac Sim as global variables
"""
parser = get_argparser()

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# Parse the arguments
args_cli = parser.parse_args()

# Launch Omniverse
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import the necessary modules after Isaac Sim is launched
from omni.isaac.orbit_tasks.utils import parse_env_cfg


# Parse the arguments
# Environment configuration
env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)

# Create the environment
env = gym.make(args_cli.task, cfg=env_cfg)

# Create the hyperparameters object
hyperparameters = Hyperparameters()

# Create the agent
agent = PPO(env, hyperparameters)
agent.policy_path = 'policies'

# Learn
# agent.learn(max_steps=2000)
agent.simulate('policies/ppo_actor_critic.pth')
```