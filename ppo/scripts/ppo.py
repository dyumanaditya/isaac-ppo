import torch

from ppo.scripts.policy.actor import Actor
from ppo.scripts.policy.critic import Critic
from ppo.scripts.memory import Memory


class PPO:
	def __init__(self, hyperparameters, env):
		self.hyperparameters = hyperparameters
		self.env = env
		self.render = self.hyperparameters.render

		# Set the device
		gpu = 0
		self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
		print(f"Using device: {self.device}")

		self.observation_space = env.observation_space.shape[0]
		self.action_space = env.action_space.shape[0]

		# Make hyperparameters class attributes
		self.gamma = self.hyperparameters.gamma
		self.lam = self.hyperparameters.lam

		self.actor_lr = self.hyperparameters.actor_lr
		self.critic_lr = self.hyperparameters.critic_lr

		self.clip_ratio = self.hyperparameters.clip_ratio
		self.kl_target = self.hyperparameters.kl_target

		self.actor_hidden_sizes = self.hyperparameters.actor_hidden_sizes
		self.critic_hidden_sizes = self.hyperparameters.critic_hidden_sizes
		self.actor_activations = self.hyperparameters.actor_activations
		self.critic_activations = self.hyperparameters.critic_activations

		self.actor_optimizer = self.hyperparameters.actor_optimizer
		self.critic_optimizer = self.hyperparameters.critic_optimizer

		# Initialize the actor and critic networks
		self.actor = Actor(self.observation_space, self.action_space, self.actor_hidden_sizes, self.actor_activations, self.device)
		self.critic = Critic(self.observation_space, self.critic_hidden_sizes, self.critic_activations, self.device)

		# Initialize the actor and critic optimizers
		self.actor_optimizer = self.actor.get_optimizer(self.actor_optimizer, self.actor_lr)
		self.critic_optimizer = self.critic.get_optimizer(self.critic_optimizer, self.critic_lr)

		# Iterations
		self.num_policy_iters = self.hyperparameters.num_policy_iters
		self.num_value_iters = self.hyperparameters.num_value_iters
		self.num_epochs = self.hyperparameters.num_epochs
		self.max_steps = self.hyperparameters.max_steps

		# Initialize memory
		self.memory_size = self.hyperparameters.memory_size
		self.memory = Memory(self.observation_space, self.action_space, self.memory_size, self.device, self.gamma, self.lam)
		self.data = None

	def learn(self, actor_model_path=None, critic_model_path=None):
		# Reset the environment
		state, _ = self.env.reset()

		# Load the model if specified
		if actor_model_path is not None:
			self.actor.load_state_dict(torch.load(actor_model_path))
		if critic_model_path is not None:
			self.critic.load_state_dict(torch.load(critic_model_path))

		# Episode related information
		episode_counter = 0
		episode_reward = 0

		for epoch in range(self.num_epochs):
			for iteration in range(self.max_steps):
				if self.render:
					self.env.render()

				# Get the action from the actor and take it
				action = self.actor.get_action(torch.as_tensor(state, dtype=torch.float32).to(self.device))
				next_state, reward, done, _, _ = self.env.step(action)
				episode_reward += reward

				# Store the transition
				self._store_transition(state, action, reward)

				# Set to new observation
				state = next_state

				# Check if the epoch ended
				epoch_ended = iteration == self.max_steps - 1

				# If the epoch ended, or we reached a terminal state in the environment
				if done or epoch_ended:
					# Calculate the value of the current state to estimate the return if the trajectory was cut halfway
					# Otherwise the terminal value is 0 (because there is no next state)
					if epoch_ended:
						value = self.critic(torch.as_tensor(state, dtype=torch.float32).to(self.device)).detach().cpu().numpy()
					else:
						value = 0

					# Finish the trajectory
					self.memory.finish_trajectory(value)

					# Print info to screen if done
					if done:
						print(f"Episode: {episode_counter}, Reward: {episode_reward}")
						# try:
						# 	print('Loss', policy_loss, value_loss)
						# except:
						# 	pass
						episode_counter += 1

					state, _ = self.env.reset()
					episode_reward = 0

			# Update the actor and critic
			policy_loss = self._update_actor()
			value_loss = self._update_critic()

			# Save the model
			if epoch % 100 == 0:
				torch.save(self.actor.state_dict(), f"policies/ppo_actor_{epoch}.pth")
				torch.save(self.critic.state_dict(), f"policies/ppo_critic_{epoch}.pth")

		# End
		self.env.close()

	def simulate(self, actor_model_path, critic_model_path):
		# Load the model
		self.actor.load_state_dict(torch.load(actor_model_path))
		self.critic.load_state_dict(torch.load(critic_model_path))

		max_episodes = 20

		# Reset the environment
		state, _ = self.env.reset()

		# Episode related information
		episode_counter = 0
		episode_reward = 0

		# Simulate the environment
		while episode_counter < max_episodes:
			if self.render:
				self.env.render()

			# Get the action from the actor and take it
			action = self.actor.get_action(torch.as_tensor(state, dtype=torch.float32).to(self.device))
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

	def _update_actor(self):
		# Get data from the memory
		self.data = self.memory.get()
		loss_pi = None
		pi_info = None

		# Train policy for multiple steps of gradient descent
		for i in range(self.num_policy_iters):
			self.actor_optimizer.zero_grad()
			loss_pi, pi_info = self._compute_actor_loss(self.data)

			# Check if the KL divergence is within the target range
			if pi_info['kl'] > 1.5 * self.kl_target:
				break

			loss_pi.backward()
			self.actor_optimizer.step()

		return loss_pi.item()

	def _update_critic(self):
		assert self.data is not None, "You need to get data from the memory by updating the actor before updating the critic"
		value_loss = None

		# Train value function for multiple steps of gradient descent
		for i in range(self.num_value_iters):
			self.critic_optimizer.zero_grad()
			value_loss = self._compute_critic_loss(self.data)
			value_loss.backward()
			self.critic_optimizer.step()

		# Return the value loss after all iterations
		return value_loss.item()

	def _compute_actor_loss(self, data):
		states, actions, advantages, log_probs_old = data['states'], data['actions'], data['advantages'], data['log_probs']

		# Policy loss
		pi, log_probs = self.actor(states, actions)
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

		return loss_pi, pi_info

	def _compute_critic_loss(self, data):
		state, reward = data['states'], data['returns']
		return (self.critic(state) - reward).pow(2).mean()

	def _store_transition(self, state, action, reward):
		value = self.critic(torch.as_tensor(state, dtype=torch.float32).to(self.device)).detach().cpu().numpy()
		log_prob = self.actor.log_prob_from_distribution(torch.as_tensor(state, dtype=torch.float32).to(self.device),
														 torch.as_tensor(action, dtype=torch.float32).to(self.device)).detach().cpu().numpy()
		self.memory.store_transition(state, action, reward, value, log_prob)
