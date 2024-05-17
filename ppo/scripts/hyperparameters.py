class Hyperparameters:
	"""
		Contains the default hyperparameters for the PPO algorithm
	"""
	def __init__(self):
		self.gamma = 0.99
		self.lam = 0.95
		self.actor_lr = 3e-4
		self.critic_lr = 3e-4

		self.clip_ratio = 0.2
		self.kl_target = 0.01

		self.actor_hidden_sizes = [64, 64]
		self.critic_hidden_sizes = [64, 64]
		self.actor_activations = ['tanh', 'tanh', 'none']
		self.critic_activations = ['tanh', 'tanh', 'none']

		self.actor_optimizer = 'Adam'
		self.critic_optimizer = 'Adam'

		self.memory_size = 1700

		self.num_policy_iters = 10
		self.num_value_iters = 80

		self.num_epochs = 3000
		self.max_steps = 1700

		self.render = False
