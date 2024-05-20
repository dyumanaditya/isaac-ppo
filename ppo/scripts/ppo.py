import torch
import numpy as np

from ppo.scripts.memory import Memory
from ppo.scripts.policy.actor_critic import ActorCritic
from ppo.scripts.hyperparameters import Hyperparameters


class PPO:
	def __init__(self, env, hyperparameters=None):
		self.hyperparameters = hyperparameters
		self.env = env
		self.render = self.hyperparameters.render

		if self.hyperparameters is None:
			self.hyperparameters = Hyperparameters()

		# Set the device
		gpu = 0
		self.compute = self.hyperparameters.device
		if self.compute == 'gpu':
			self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
		else:
			self.device = torch.device("cpu")
		print(f"Using device: {self.device}")

		self.observation_space = env.observation_space.shape[0]
		self.action_space = env.action_space.shape[0]

		# Make hyperparameters class attributes
		self.gamma = self.hyperparameters.gamma
		self.lam = self.hyperparameters.lam

		self.lr = self.hyperparameters.lr

		self.clip_ratio = self.hyperparameters.clip_ratio
		self.kl_target = self.hyperparameters.kl_target

		self.value_loss_coef = self.hyperparameters.value_loss_coef
		self.entropy_coef = self.hyperparameters.entropy_coef
		self.max_grad_norm = self.hyperparameters.max_grad_norm

		self.actor_hidden_sizes = self.hyperparameters.actor_hidden_sizes
		self.critic_hidden_sizes = self.hyperparameters.critic_hidden_sizes
		self.actor_activations = self.hyperparameters.actor_activations
		self.critic_activations = self.hyperparameters.critic_activations

		self.optimizer = self.hyperparameters.optimizer

		# Initialize the actor and critic networks
		self.actor_critic = ActorCritic(self.observation_space, self.action_space, self.actor_hidden_sizes, self.critic_hidden_sizes, self.actor_activations, self.critic_activations, self.device).to(self.device)

		# Initialize the actor and critic optimizers
		self.actor_critic_optimizer = self.actor_critic.get_optimizer(self.optimizer, self.lr)

		# Iterations
		self.num_epochs = self.hyperparameters.num_epochs
		self.minibatch_size = self.hyperparameters.minibatch_size

		# Initialize memory
		self.memory_size = self.hyperparameters.memory_size
		self.memory = Memory(self.observation_space, self.action_space, self.memory_size, self.device, self.gamma, self.lam)
		self.normalize_advantages = self.hyperparameters.normalize_advantages

	def learn(self, max_steps=100000, actor_critic_model_path=None):
		# Reset the environment
		state, _ = self.env.reset()
		loss = 0

		# Load the model if specified
		if actor_critic_model_path is not None:
			self.actor_critic.load_state_dict(torch.load(actor_critic_model_path))

		# Episode related information
		episode_counter = 0
		episode_reward = 0

		for timestep in range(max_steps):
			# Collect rollouts
			rollout_episode_counter = 0
			max_rollout_reward = -np.inf
			for rollout in range(self.memory_size):
				if self.render:
					self.env.render()

				# Get the action from the actor and take it
				action = self.actor_critic.get_action(torch.as_tensor(state, dtype=torch.float32).to(self.device))
				next_state, reward, done, truncated, _ = self.env.step(action)
				episode_reward += reward

				# Store the transition
				self._store_transition(state, action, reward)

				# Set to new observation
				state = next_state

				# Check if the epoch ended
				rollout_ended = rollout == self.memory_size - 1

				# If the epoch ended, or we reached a terminal state in the environment
				if done or truncated or rollout_ended:
					# Calculate the value of the current state to estimate the return if the trajectory was cut halfway
					# Otherwise the terminal value is 0 (because there is no next state)
					if rollout_ended or truncated:
						value = self.actor_critic.get_value(torch.as_tensor(state, dtype=torch.float32).to(self.device)).detach().cpu().numpy()
					else:
						value = 0

					# Finish the trajectory
					self.memory.finish_trajectory(value)

					state, _ = self.env.reset()
					max_rollout_reward = max(max_rollout_reward, episode_reward)
					episode_reward = 0
					episode_counter += 1
					rollout_episode_counter += 1

			# Print the mean reward of the last episodes in rollout
			if timestep % 1 == 0 and rollout_episode_counter != 0:
				print(f"Mean episode returns: {round(np.sum(self.memory.rewards) / rollout_episode_counter, 3)}, Loss: {round(loss, 6)} Episode: {episode_counter}, Max Episode Reward: {max_rollout_reward}", flush=True)

			# Go over the rollouts for multiple epochs
			for epoch in range(self.num_epochs):
				# print(epoch)
				# Go over each of the mini batches
				for rollout_minibatch in self.memory.get_minibatches(self.minibatch_size):
					# Update the actor and critic
					loss = self._update_actor_critic(rollout_minibatch)

			# Reset the memory
			self.memory.reset()

			# Save the model
			if timestep % 100 == 0:
				torch.save(self.actor_critic.state_dict(), f"policies/ppo_actor_critic_{timestep}.pth")

		# End
		self.env.close()

	def simulate(self, actor_critic_model_path, max_episodes=20):
		# Load the model
		self.actor_critic.load_state_dict(torch.load(actor_critic_model_path))

		# Reset the environment
		state, _ = self.env.reset()

		# Episode related information
		episode_counter = 0
		episode_reward = 0

		# Simulate the environment
		while episode_counter <= max_episodes:
			if self.render:
				self.env.render()

			# Get the action from the actor and take it
			action = self.actor_critic.get_action(torch.as_tensor(state, dtype=torch.float32).to(self.device))
			next_state, reward, done, _, _ = self.env.step(action)
			episode_reward += reward

			# Set to new observation
			state = next_state

			# Print info to screen if done
			if done:
				print(f"Episode: {episode_counter}, Reward: {episode_reward}")
				episode_counter += 1

				state, _ = self.env.reset()
				episode_reward = 0

		# End
		self.env.close()

	def save(self, path):
		torch.save(self.actor_critic.state_dict(), path + '.pth')

	def load(self, path):
		self.actor_critic.load_state_dict(torch.load(path + '.pth'))

	def _update_actor_critic(self, minibatch):
		# Gather losses
		loss, pi_info = self._compute_loss(minibatch)

		# Perform optimization with clipped gradients
		if self.kl_target is not None and pi_info['kl'] > 1.5 * self.kl_target:
			return 0
		else:
			self.actor_critic_optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
			self.actor_critic_optimizer.step()

			return loss.item()

	def _compute_loss(self, data):
		states, actions, advantages, log_probs_old, returns = data['states'], data['actions'], data['advantages'], data['log_probs'], data['returns']

		# Normalize advantages
		if self.normalize_advantages:
			assert self.minibatch_size > 1, "Minibatch size must be greater than 1 to normalize advantages"
			advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

		# Policy loss
		pi, log_probs, value = self.actor_critic(states, actions)
		ratio = torch.exp(log_probs - log_probs_old)
		clip_advantages = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
		loss_pi = -torch.min(ratio * advantages, clip_advantages).mean()

		# Other useful values
		# Clipped is a boolean mask that corresponds to the values that were clipped
		approx_kl = (log_probs_old - log_probs).mean().item()
		entropy = pi.entropy().mean().item()
		clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
		clip_frac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
		pi_info = dict(kl=approx_kl, entropy=entropy, cf=clip_frac)

		# Value Loss
		value_loss = (value - returns).pow(2).mean()

		# Clip value loss
		value_clipped = value + (value - returns).clamp(-self.clip_ratio, self.clip_ratio)
		value_loss_clipped = (value_clipped - returns).pow(2).mean()
		value_loss = torch.max(value_loss, value_loss_clipped)

		# Combine losses
		loss = loss_pi + self.value_loss_coef * value_loss - self.entropy_coef * entropy

		return loss, pi_info

	def _store_transition(self, state, action, reward):
		value = self.actor_critic.get_value(torch.as_tensor(state, dtype=torch.float32).to(self.device)).detach().cpu().numpy()
		log_prob = self.actor_critic.log_prob_from_distribution(torch.as_tensor(state, dtype=torch.float32).to(self.device),
																torch.as_tensor(action, dtype=torch.float32).to(self.device)).detach().cpu().numpy()
		self.memory.store_transition(state, action, reward, value, log_prob)
