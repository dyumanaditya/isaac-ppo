import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
	def __init__(self, input_dim, output_dim, hidden_sizes, activations, device):
		super(MLP, self).__init__()
		self.device = device

		# Create a dictionary of activation functions
		self.activations_dict = nn.ModuleDict({
			"relu": nn.ReLU(),
			"tanh": nn.Tanh(),
			"sigmoid": nn.Sigmoid(),
			"softmax": nn.LogSoftmax(),
			"selu": nn.SELU(),
			"elu": nn.ELU(),
			"leakyrelu": nn.LeakyReLU(),
			"none": nn.Identity(),
		})

		layers = []

		# Create the input layer
		layers.append(nn.Linear(input_dim, hidden_sizes[0]))
		layers.append(self.activations_dict[activations[0]])

		# Create the hidden layers
		for i in range(len(hidden_sizes) - 1):
			layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
			layers.append(self.activations_dict[activations[i + 1]])

		# Create the output layer
		layers.append(nn.Linear(hidden_sizes[-1], output_dim))

		# Create the network
		self.mlp = nn.Sequential(*layers).to(device)

	def forward(self, x):
		# Convert observation to tensor if it is a numpy array
		if isinstance(x, np.ndarray):
			x = torch.tensor(x, dtype=torch.float).to(self.device)
		return self.mlp(x)
