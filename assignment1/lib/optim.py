from __future__ import print_function, division
from future import standard_library
import numpy as np


""" Super Class """
class Optimizer(object):
	""" 
	This is a template for implementing the classes of optimizers
	"""
	def __init__(self, net, lr=1e-4):
		self.net = net  # the model
		self.lr = lr    # learning rate

	""" Make a step and update all parameters """
	def step(self):
		for layer in self.net.layers:
			for n, v in list(layer.params.items()):
				pass


""" Classes """
class SGD(Optimizer):
	""" Some comments """
	def __init__(self, net, lr=1e-4):
		self.net = net
		self.lr = lr

	def step(self):
		for layer in self.net.layers:
			for n, v in list(layer.params.items()):
				dv = layer.grads[n]
				layer.params[n] -= self.lr * dv


class SGDM(Optimizer):
	def __init__(self, net, lr=1e-4, momentum=0.0):
		self.net = net
		self.lr = lr
		self.momentum = momentum
		self.velocity = {}

	def step(self):
		#############################################################################
		# TODO: Implement the SGD + Momentum                                        #
		#############################################################################
		for layer in self.net.layers:
			for n, v in list(layer.params.items()):
				updated_w = None
				dw = layer.grads[n]
				updated_velocity = np.zeros_like(layer.params[n])
				#print(bool(self.velocity))
				if bool(self.velocity and n in self.velocity):
					updated_velocity = self.momentum*self.velocity[n] - self.lr*dw
				else:
					updated_velocity = self.momentum * updated_velocity - self.lr * dw
				updated_w = layer.params[n] + updated_velocity # w + v
				self.velocity[n] = updated_velocity
				layer.params[n] = updated_w
		#############################################################################
		#                             END OF YOUR CODE                              #
		#############################################################################
		

class RMSProp(Optimizer):
	def __init__(self, net, lr=1e-2, decay=0.99, eps=1e-8):
		self.net = net
		self.lr = lr
		self.decay = decay
		self.eps = eps
		self.cache = {}  # decaying average of past squared gradients

	def step(self):
		#############################################################################
		# TODO: Implement the RMSProp                                               #
		#############################################################################
		for layer in self.net.layers:
			for n, v in list(layer.params.items()):
				updated_w = None
				cache = np.zeros_like(layer.params[n])
				dw = layer.grads[n]
				if bool(self.cache and n in self.cache):
					self.cache[n] = self.decay * self.cache[n] + (1 - self.decay) * dw **2
				else:
					self.cache[n] = self.decay * cache + (1 - self.decay) * dw ** 2
				updated_w = layer.params[n] - self.lr * dw /  (np.sqrt(self.cache[n]) + self.eps)
				layer.params[n] = updated_w
		#############################################################################
		#                             END OF YOUR CODE                              #
		#############################################################################


class Adam(Optimizer):
	def __init__(self, net, lr=1e-3, beta1=0.9, beta2=0.999, t=0, eps=1e-8):
		self.net = net
		self.lr = lr
		self.beta1, self.beta2 = beta1, beta2
		self.eps = eps
		self.mt = {}
		self.vt = {}
		self.t = t

	def step(self):
		#############################################################################
		# TODO: Implement the Adam                                                  #
		#############################################################################
		for layer in self.net.layers:
			for n, v in list(layer.params.items()):
				updated_w = None
				dw = layer.grads[n]
				m_default = np.zeros_like(layer.params[n])
				v_default = np.zeros_like(layer.params[n])
				self.t += 1
				if bool(self.mt and n in self.mt):
					self.mt[n] = self.beta1 * self.mt[n] + (1 - self.beta1) * dw
				else:
					self.mt[n] = self.beta1 * m_default + (1 - self.beta1) * dw
				if bool(self.vt and n in self.vt):
					self.vt[n] = self.beta2 * self.vt[n] + (1 - self.beta2) * (dw**2)
				else:
					self.vt[n] = self.beta2 * v_default + (1 - self.beta2) * (dw ** 2)
				mt = self.mt[n] / (1 - self.beta1**self.t)
				vt = self.vt[n] / (1 - self.beta2**self.t)
				updated_w = layer.params[n] - self.lr * mt / (np.sqrt(vt) + self.eps)
				layer.params[n] = updated_w
		#############################################################################
		#                             END OF YOUR CODE                              #
		#############################################################################