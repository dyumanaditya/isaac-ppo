import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def main():
	vec_env = make_vec_env("BipedalWalker-v3", n_envs=1)
	# env = gym.make('BipedalWalker-v3')
	model = PPO('MlpPolicy', vec_env, verbose=1, device='cpu')
	model.learn(total_timesteps=2560000)
	# model.save("ppo_bipedalwalker")

	# model = PPO.load("ppo_bipedalwalker")
	#
	# obs = vec_env.reset()
	# while True:
	# 	action, _states = model.predict(obs)
	# 	obs, rewards, dones, info = vec_env.step(action)
	# 	vec_env.render("human")

if __name__ == '__main__':
	main()