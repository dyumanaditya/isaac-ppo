import torch
import torch.nn as nn
from torch.distributions import Normal

from ppo.scripts.policy.network import MLP


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_sizes, activations, device):
		super(Actor, self).__init__()
		self.device = device

		# Initialize the log standard deviation to -0.5 to encourage exploration
		log_std = -0.5 * torch.ones(action_dim, dtype=torch.float)
		self.log_std = nn.Parameter(torch.as_tensor(log_std).to(self.device))

		# Initialize the actor network
		self.pi_net = MLP(state_dim, action_dim, hidden_sizes, activations).to(self.device)

	def forward(self, state, action=None):
		pi = self._distribution(state)
		log_prob = None

		if action is not None:
			log_prob = self.log_prob_from_distribution(state, action)

		return pi, log_prob

	def _distribution(self, state):
		# Get the mean from the actor network
		mean = self.pi_net.mlp(state)
		std = torch.exp(self.log_std)
		return Normal(mean, std)

	def log_prob_from_distribution(self, state, act):
		pi = self._distribution(state)
		return pi.log_prob(act).sum(axis=-1)

	def get_action(self, state):
		with torch.no_grad():
			pi = self._distribution(state)
			action = pi.sample()
		return action.cpu().numpy()

	def get_optimizer(self, optimizer, lr):
		if optimizer == 'Adam':
			return torch.optim.Adam(self.parameters(), lr=lr)
		elif optimizer == 'RMSprop':
			return torch.optim.RMSprop(self.parameters(), lr=lr)
		elif optimizer == 'SGD':
			return torch.optim.SGD(self.parameters(), lr=lr)
		else:
			raise ValueError("Invalid Actor optimizer")
