
import gym
import torch 
from torch import nn, tanh, relu
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal, Categorical
import numpy as np

class DeepSet(nn.Module):

	def __init__(self, state_dim, latent_dim):
		super(DeepSet, self).__init__()

		self.input_dim = state_dim
		self.output_dim = latent_dim

		phi_layers = nn.ModuleList([
			nn.Linear(self.input_dim,64),
			nn.Linear(64,64),
			nn.Linear(64,16)])
		rho_layers = nn.ModuleList([
		nn.Linear(16,64),
		nn.Linear(64,64),
		nn.Linear(64,self.output_dim)])
		activation = relu
		
		self.phi = FeedForward(phi_layers,activation)
		self.rho = FeedForward(rho_layers,activation)
		self.device = torch.device('cpu')

	def to(self, device):
		self.device = device
		self.phi.to(device)
		self.rho.to(device)
		return super().to(device)

	def export_to_onnx(self, filename):
		self.phi.export_to_onnx("{}_phi".format(filename))
		self.rho.export_to_onnx("{}_rho".format(filename))

	def forward(self,x):
		X = torch.zeros((len(x),self.rho.in_dim), device=self.device)
		num_elements = int(x.size()[1] / self.phi.in_dim)
		for i in range(num_elements):
			X += self.phi(x[:,i*self.phi.in_dim:(i+1)*self.phi.in_dim])
		return self.rho(X)

class FeedForward(nn.Module):

	def __init__(self,layers,activation):
		super(FeedForward, self).__init__()
		self.layers = layers
		self.activation = activation

		self.in_dim = layers[0].in_features
		self.out_dim = layers[-1].out_features

	def forward(self, x):
		for layer in self.layers[:-1]:
			x = self.activation(layer(x))
		x = self.layers[-1](x)
		return x

	def export_to_onnx(self, filename):
		dummy_input = torch.randn(self.in_dim)
		torch.onnx.export(self, dummy_input, "{}.onnx".format(filename), export_params=True, keep_initializers_as_inputs=True)