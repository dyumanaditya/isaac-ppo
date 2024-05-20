class Hyperparameters:
	"""
		Contains the default hyperparameters for the PPO algorithm
	"""
	def __init__(self):
		self.device = 'gpu'
		self.gamma = 0.99
		self.lam = 0.95
		self.lr = 1e-4

		self.clip_ratio = 0.2
		# self.kl_target = 0.01
		self.kl_target = None

		self.value_loss_coef = 0.5
		self.entropy_coef = 0.0
		self.max_grad_norm = 0.5

		self.actor_hidden_sizes = [64, 64]
		self.critic_hidden_sizes = [64, 64]
		self.actor_activations = ['tanh', 'tanh', 'none']
		self.critic_activations = ['tanh', 'tanh', 'none']

		self.optimizer = 'Adam'

		self.memory_size = 2048
		# self.memory_size = 4096

		self.num_epochs = 10
		self.minibatch_size = 64
		self.normalize_advantages = True

		self.render = False
