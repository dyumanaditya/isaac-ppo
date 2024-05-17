import gym

from ppo.scripts.ppo import PPO
from ppo.scripts.hyperparameters import Hyperparameters


def main():
	env = gym.make("BipedalWalker-v3", render_mode=None)

	# Create the hyperparameters object
	hyperparameters = Hyperparameters()
	hyperparameters.render = True

	agent = PPO(hyperparameters, env)
	agent.learn()
	# agent.simulate('policies/ppo_actor_800.pth', 'policies/ppo_critic_800.pth')


if __name__ == '__main__':
	main()
