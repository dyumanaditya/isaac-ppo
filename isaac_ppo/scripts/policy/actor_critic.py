import torch
from torch import nn

from isaac_ppo.scripts.policy.actor import Actor
from isaac_ppo.scripts.policy.critic import Critic


class ActorCritic(nn.Module):
	def __init__(self, state_dim, action_dim, actor_hidden_sizes, critic_hidden_sizes, actor_activations, critic_activations, device):
		super(ActorCritic, self).__init__()
		self.device = device

		# Create Actor and Critic
		self.actor = Actor(state_dim, action_dim, actor_hidden_sizes, actor_activations, device).to(device)
		self.critic = Critic(state_dim, critic_hidden_sizes, critic_activations, device).to(device)

	def forward(self, state, action=None):
		pi, log_prob = self.actor(state, action)
		value = self.critic(state)
		return pi, log_prob, value

	def get_action(self, state):
		return self.actor.get_action(state)

	def get_value(self, state):
		return self.critic(state)

	def log_prob_from_distribution(self, act):
		return self.actor.log_prob_from_distribution(act)

	def get_mu_sigma(self):
		return self.actor.get_mu_sigma()

	def get_entropy(self):
		return self.actor.distribution.entropy().sum(dim=-1)


	def get_optimizer(self, optimizer, lr):
		if optimizer == 'Adam':
			return torch.optim.Adam(self.parameters(), lr=lr)
		elif optimizer == 'RMSprop':
			return torch.optim.RMSprop(self.parameters(), lr=lr)
		elif optimizer == 'SGD':
			return torch.optim.SGD(self.parameters(), lr=lr)
		else:
			raise ValueError("Invalid optimizer")
