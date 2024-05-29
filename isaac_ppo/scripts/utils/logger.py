import os
import torch
import json
import pandas as pd


class Logger:
	"""
	Logger class to log data to environment and training specific folders.
	1. logs: Contains the training logs csv files
	2. policies: Contains the saved models
	3. videos: Contains the videos of the training
	"""

	def __init__(self, log_dir, stamp, save_freq=20, record_video=False):
		"""
		Constructor for the Logger class
		:param log_dir: Directory to save all logs for a particular run
		:param stamp: unique stamp for the experiment
		:param save_freq: Frequency at which to save and clear the CSV logs (in timesteps)
		:param record_video: Whether to record videos of the training
		"""
		self.log_dir = log_dir
		self.stamp = stamp
		self.save_freq = save_freq
		self.record_video = record_video
		self.log_root_folder = str(os.path.join(log_dir, stamp))

		# Create folders for saved policies, logs, hyperparameters, videos
		self.policy_folder = os.path.join(self.log_root_folder, 'policies')
		self.log_folder = os.path.join(self.log_root_folder, 'logs')
		self.hyperparameters_folder = os.path.join(self.log_root_folder, 'hyperparameters')
		self.video_folder = os.path.join(self.log_root_folder, 'videos')

		# Create the folders if they don't exist
		os.makedirs(self.policy_folder, exist_ok=True)
		os.makedirs(self.log_folder, exist_ok=True)
		os.makedirs(self.hyperparameters_folder, exist_ok=True)
		if self.record_video:
			os.makedirs(self.video_folder, exist_ok=True)

		# There is only one log file containing all the training information in CSV format
		self.log_file = os.path.join(self.log_folder, 'training_log.csv')
		self.log_header = ['total_steps', 'total_env_steps', 'total_episodes', 'total_policy_updates',
						   'mean_episode_reward',
						   'mean_episode_length', 'mean_loss', 'mean_value_loss', 'mean_surrogate_loss',
						   'learning_rate']
		self._reset_log_buffer()

		# Open file to save headers
		self.log_buffer.to_csv(self.log_file, mode='a', header=True, index=False)

	def log_policy(self, policy_state_dict, optimizer_state_dict, timestep):
		"""
		Save the policy to the policies folder
		:param policy_state_dict: The policy state dictionary
		:param optimizer_state_dict: The optimizer state dictionary
		:param timestep: The timestep at which the policy was saved
		"""
		if timestep % self.save_freq == 0:
			policy_path = os.path.join(self.policy_folder, f'policy_{timestep}.pth')
			optimizer_path = os.path.join(self.policy_folder, f'optimizer_{timestep}.pth')
			torch.save(policy_state_dict, policy_path)
			torch.save(optimizer_state_dict, optimizer_path)

	def log_training_info(self, total_steps, total_env_steps, total_episodes, total_policy_updates, mean_episode_reward,
						  mean_episode_length, mean_loss, mean_value_loss, mean_surrogate_loss, learning_rate):

		# Save the training information to the log buffer
		self.log_buffer = pd.concat(
			[self.log_buffer, pd.DataFrame({'total_steps': [total_steps], 'total_env_steps': [total_env_steps],
											'total_episodes': [total_episodes],
											'total_policy_updates': [total_policy_updates],
											'mean_episode_reward': [mean_episode_reward],
											'mean_episode_length': [mean_episode_length],
											'mean_loss': [mean_loss], 'mean_value_loss': [mean_value_loss],
											'mean_surrogate_loss': [mean_surrogate_loss],
											'learning_rate': [learning_rate]})], ignore_index=True)

		# If the total steps are a multiple of the save frequency, save the training information
		if total_steps % self.save_freq == 0:
			self.log_buffer.to_csv(self.log_file, mode='a', header=False, index=False)
			self._reset_log_buffer()

	@staticmethod
	def log_to_console(total_steps, total_env_steps, total_episodes, total_policy_updates, mean_episode_reward, mean_episode_length,
					   mean_loss, mean_value_loss, mean_surrogate_loss, learning_rate):
		print("-" * 50)
		print(f"Total steps: {total_steps}", flush=True)
		print(f"Total env steps: {total_env_steps}", flush=True)
		print(f"Total episodes: {total_episodes}", flush=True)
		print(f"Total policy updates: {total_policy_updates}", flush=True)
		print("--")
		print(f"Mean Episode Reward: {mean_episode_reward}", flush=True)
		print(f"Mean Episode Length: {mean_episode_length}", flush=True)
		print("--")
		print(f"Mean Loss: {mean_loss}", flush=True)
		print(f"Mean Value Loss: {mean_value_loss}", flush=True)
		print(f"Mean Surrogate Loss: {mean_surrogate_loss}", flush=True)
		print("--")
		print(f"Learning Rate: {learning_rate}", flush=True)
		print("-" * 50)

	def log_hyperparameters(self, hyperparameters_dict):
		"""
		Save the hyperparameters to the hyperparameters folder
		:param hyperparameters_dict: Dictionary containing the hyperparameters
		"""
		hyperparameters_path = os.path.join(self.hyperparameters_folder, 'hyperparameters.json')
		with open(hyperparameters_path, 'w') as f:
			json.dump(hyperparameters_dict, f, indent=4)

	def _reset_log_buffer(self):
		buffers = {header: [] for header in self.log_header}
		self.log_buffer = pd.DataFrame(buffers)
