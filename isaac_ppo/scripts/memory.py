import numpy as np
import scipy
import torch


class Memory:
	def __init__(self, state_dim, action_dim, num_envs, num_transitions_per_env, device, gamma=0.99, lam=0.95):
		self.states = torch.zeros((num_transitions_per_env, num_envs, state_dim), dtype=torch.float32, device=device)
		self.actions = torch.zeros((num_transitions_per_env, num_envs, action_dim), dtype=torch.float32, device=device)
		self.advantages = torch.zeros((num_transitions_per_env, num_envs, 1), dtype=torch.float32, device=device)
		self.rewards = torch.zeros((num_transitions_per_env, num_envs, 1), dtype=torch.float32, device=device)
		self.returns = torch.zeros((num_transitions_per_env, num_envs, 1), dtype=torch.float32, device=device)
		self.dones = torch.zeros((num_transitions_per_env, num_envs, 1), dtype=torch.float32, device=device).byte()
		self.values = torch.zeros((num_transitions_per_env, num_envs, 1), dtype=torch.float32, device=device)
		self.log_probs = torch.zeros((num_transitions_per_env, num_envs, 1), dtype=torch.float32, device=device)
		self.mu = torch.zeros((num_transitions_per_env, num_envs, action_dim), dtype=torch.float32, device=device)
		self.sigma = torch.zeros((num_transitions_per_env, num_envs, action_dim), dtype=torch.float32, device=device)
		self.gamma = gamma
		self.lam = lam
		self.device = device

		self.step = 0
		self.num_envs = num_envs
		self.num_transitions_per_env = num_transitions_per_env

	def store_transitions(self, states, actions, rewards, dones, values, log_probs, actions_mu, actions_sigma):
		# Buffer has to have room so you can store
		assert self.step < self.num_transitions_per_env, 'Buffer is full, you cannot store more transitions'
		self.states[self.step].copy_(states)
		self.actions[self.step].copy_(actions)
		self.rewards[self.step].copy_(rewards.view(-1, 1))
		self.dones[self.step].copy_(dones.view(-1, 1))
		self.values[self.step].copy_(values.view(-1, 1))
		self.log_probs[self.step].copy_(log_probs.view(-1, 1))
		self.mu[self.step].copy_(actions_mu)
		self.sigma[self.step].copy_(actions_sigma)
		self.step += 1

	def compute_returns(self, last_values):
		# GAE
		advantage = 0
		for step in reversed(range(self.num_transitions_per_env)):
			if step == self.num_transitions_per_env - 1:
				next_values = last_values
			else:
				next_values = self.values[step + 1]

			next_is_not_terminal = 1.0 - self.dones[step].float()
			delta = self.rewards[step] + next_is_not_terminal * self.gamma * next_values - self.values[step]
			advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
			self.returns[step] = advantage + self.values[step]

		# Compute and normalize the advantages
		self.advantages = self.returns - self.values
		self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

	def get_minibatches(self, num_minibatches):
		# Buffer has to be full before you can get
		assert self.step == self.num_transitions_per_env, "Buffer is not full, you cannot get from it."

		batch_size = self.num_envs * self.num_transitions_per_env
		minibatch_size = batch_size // num_minibatches

		# Shuffle the data
		indices = torch.randperm(minibatch_size * num_minibatches, requires_grad=False, device=self.device)

		# Change dimensions of the transition values
		states = self.states.flatten(0, 1)
		actions = self.actions.flatten(0, 1)
		returns = self.returns.flatten(0, 1)
		values = self.values.flatten(0, 1)
		advantages = self.advantages.flatten(0, 1)
		log_probs = self.log_probs.flatten(0, 1)
		mu = self.mu.flatten(0, 1)
		sigma = self.sigma.flatten(0, 1)

		for i in range(num_minibatches):
			start = i * minibatch_size
			end = (i + 1) * minibatch_size
			batch_indices = indices[start:end]

			states_batch = states[batch_indices]
			actions_batch = actions[batch_indices]
			returns_batch = returns[batch_indices]
			values_batch = values[batch_indices]
			advantages_batch = advantages[batch_indices]
			log_probs_batch = log_probs[batch_indices]
			mu_batch = mu[batch_indices]
			sigma_batch = sigma[batch_indices]
			yield states_batch, actions_batch, returns_batch, values_batch, advantages_batch, log_probs_batch, mu_batch, sigma_batch

	def reset(self):
		self.step = 0
