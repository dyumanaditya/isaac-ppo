import torch
import torch.nn as nn
from torch.distributions import Normal
from torchsummary import summary

from isaac_ppo.scripts.policy.network import MLP


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_sizes, activations, device):
		super(Actor, self).__init__()
		self.device = device

		# Store these variables so that we don't have to do a forward pass multiple times
		self.distribution = None

		# Initialize the log standard deviation to -0.5 to encourage exploration
		std = 1.0 * torch.ones(action_dim, dtype=torch.float)
		self.std = nn.Parameter(torch.as_tensor(std).to(self.device))

		# Initialize the actor network
		self.pi_net = MLP(state_dim, action_dim, hidden_sizes, activations, device).to(self.device)
		print("Actor Network")
		print(summary(self.pi_net.mlp, (state_dim,)))
		print(self.pi_net.mlp, '\n')

	def forward(self, state, action=None):
		pi = self._distribution(state)
		log_prob = None

		if action is not None:
			log_prob = self.log_prob_from_distribution(action)

		return pi, log_prob

	def _distribution(self, state):
		# Get the mean from the actor network
		# Update the distribution
		mean = self.pi_net.mlp(state)
		self.distribution = Normal(mean, self.std)
		return self.distribution

	def log_prob_from_distribution(self, act):
		return self.distribution.log_prob(act).sum(dim=-1)

	def get_action(self, state):
		pi = self._distribution(state)
		action = pi.sample()
		return action

	def get_mu_sigma(self):
		return self.distribution.mean, self.distribution.stddev

	def get_optimizer(self, optimizer, lr):
		if optimizer == 'Adam':
			return torch.optim.Adam(self.parameters(), lr=lr)
		elif optimizer == 'RMSprop':
			return torch.optim.RMSprop(self.parameters(), lr=lr)
		elif optimizer == 'SGD':
			return torch.optim.SGD(self.parameters(), lr=lr)
		else:
			raise ValueError("Invalid Actor optimizer")
