class Hyperparameters:
	"""
		Contains the default hyperparameters for the PPO algorithm
	"""
	def __init__(self):
		self.gamma = 0.99
		self.lam = 0.95
		self.lr = 1e-3

		self.clip_ratio = 0.2
		self.kl_target = 0.01
		# self.kl_target = None

		self.value_loss_coef = 1.0
		self.entropy_coef = 0.01
		self.max_grad_norm = 1.0
		self.clip_value_loss = True

		self.actor_hidden_sizes = [512, 256, 128]
		self.critic_hidden_sizes = [512, 256, 128]
		self.actor_activations = ['elu', 'elu', 'elu']
		self.critic_activations = ['elu', 'elu', 'elu']

		self.optimizer = 'Adam'

		# self.num_transitions_per_env = 96
		self.num_transitions_per_env = 24

		self.num_epochs = 5
		self.num_minibatches = 4

	def as_dict(self):
		return self.__dict__
