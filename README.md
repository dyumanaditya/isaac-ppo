# Proximal Policy Optimization
This repository contains an implementation of the RL Algorithm Proximal Policy Optimization.
The implementation is based on the paper [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) by Schulman et al.
and is inspired from [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3).


## Installation
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
import gym
from ppo import PPO, Hyperparameters

env = gym.make("BipedalWalker-v3", render_mode=None)

# Modify any of the default hyperparameters
hyperparameters = Hyperparameters()
hyperparameters.lr = 0.0001

model = PPO(env, hyperparameters)
model.learn(max_steps=100000)
model.save("BipedalWalker-v3")

```

### Evaluation
```python
import gym
from ppo import PPO

env = gym.make("BipedalWalker-v3", render_mode="human")
model = PPO(env)
model.simulate("BipedalWalker-v3", max_episodes=10)
```