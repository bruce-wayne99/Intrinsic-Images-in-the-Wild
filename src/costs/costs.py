import numpy as np

class CostFunction():
	def __init__(self, input, params):
		self.input = input
		self.params = params
		self.features = self.get_features()

	def cost_r(self, rm):
		return 0

	def cost_s(self, sm):
		return 0

	def compute_unary_costs(self, intensities, chromaticities):
		nlabels = intensities.shape[0]
		unary_costs = np.zeros((self.input.mask_nnz, nlabels), dtype=np.float32)
		return unary_costs

	def compute_pairwise_costs(self, intensities, chromaticities):
		nlabels = intensities.shape[0]
		binary_costs = np.zeros((nlabels, nlabels), dtype=np.float32)
		return binary_costs

	def get_features(self):
		mask_nz = self.input.mask_nz
		mask_nnz = self.input.mask_nnz
		features = np.zeros((mask_nnz, 5), dtype=np.float32)
		# intensity
		features[:, 0] = self.input.img_irg[mask_nz[0], mask_nz[1], 0] / self.params.theta_l
		# chromaticity
		features[:, 1] = self.input.img_irg[mask_nz[0], mask_nz[1], 1] / self.params.theta_c
		features[:, 2] = self.input.img_irg[mask_nz[0], mask_nz[1], 2] / self.params.theta_c
		# pixel location
		features[:, 3] = mask_nz[0] / (self.params.theta_p * self.input.diag)
		features[:, 4] = mask_nz[1] / (self.params.theta_p * self.input.diag)
		return features