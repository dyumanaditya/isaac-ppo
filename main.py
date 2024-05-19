import gym

from ppo.scripts.ppo import PPO
from ppo.scripts.hyperparameters import Hyperparameters


def main():
	env_name = 'BipedalWalker-v3'
	render = False
	learn = True

	if not learn:
		render = True

	if render:
		env = gym.make(env_name, render_mode='human')
	else:
		env = gym.make(env_name, render_mode=None)

	# Create the hyperparameters object
	hyperparameters = Hyperparameters()
	hyperparameters.render = render

	agent = PPO(hyperparameters, env)
	if learn:
		agent.learn()
	else:
		agent.simulate('policies/ppo_actor_critic_100.pth')


if __name__ == '__main__':
	main()
