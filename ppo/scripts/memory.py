import numpy as np
import scipy
import torch


class Memory:
	def __init__(self, state_dim, action_dim, size, device, gamma=0.99, lam=0.95):
		self.states = np.zeros((size, state_dim), dtype=np.float32)
		self.actions = np.zeros((size, action_dim), dtype=np.float32)
		self.advantages = np.zeros(size, dtype=np.float32)
		self.rewards = np.zeros(size, dtype=np.float32)
		self.returns = np.zeros(size, dtype=np.float32)
		self.values = np.zeros(size, dtype=np.float32)
		self.log_probs = np.zeros(size, dtype=np.float32)
		self.gamma = gamma
		self.lam = lam
		self.device = device

		self.step = 0
		self.max_size = size
		self.trajectory_start_idx = 0

	def store_transition(self, state, action, reward, value, log_prob):
		# Buffer has to have room so you can store
		assert self.step < self.max_size, 'Buffer is full, you cannot store more transitions'
		self.states[self.step] = state
		self.actions[self.step] = action
		self.rewards[self.step] = reward
		self.values[self.step] = value
		self.log_probs[self.step] = log_prob
		self.step += 1

	def finish_trajectory(self, last_value=0):
		path_slice_idx = slice(self.trajectory_start_idx, self.step)

		rewards = np.append(self.rewards[path_slice_idx], last_value)
		values = np.append(self.values[path_slice_idx], last_value)

		# Calculate the GAE Advantage
		deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
		self.advantages[path_slice_idx] = self.discount_return(deltas, self.gamma * self.lam)

		# Computes return
		self.returns[path_slice_idx] = self.discount_return(rewards, self.gamma)[:-1]

		self.trajectory_start_idx = self.step

	def get(self):
		# Buffer has to be full before you can get
		assert self.step == self.max_size, "Buffer is not full, you cannot get from it."
		self.step = 0
		self.trajectory_start_idx = 0
		data = dict(states=self.states, actions=self.actions, returns=self.returns,
					advantages=self.advantages, log_probs=self.log_probs)
		return {k: torch.as_tensor(v, dtype=torch.float32).to(self.device) for k, v in data.items()}

	def get_minibatches(self, batch_size):
		# Buffer has to be full before you can get
		assert self.step == self.max_size, "Buffer is not full, you cannot get from it."

		# Shuffle the data
		indices = np.arange(self.max_size)
		np.random.shuffle(indices)

		for i in range(0, self.max_size, batch_size):
			idx = indices[i:i + batch_size]
			data = dict(states=self.states[idx], actions=self.actions[idx], returns=self.returns[idx],
						advantages=self.advantages[idx], log_probs=self.log_probs[idx])
			yield {k: torch.as_tensor(v, dtype=torch.float32).to(self.device) for k, v in data.items()}

	def reset(self):
		self.step = 0
		self.trajectory_start_idx = 0

	@staticmethod
	def discount_return(rewards, discount_factor):
		return scipy.signal.lfilter([1], [1, float(-discount_factor)], rewards[::-1], axis=0)[::-1]
