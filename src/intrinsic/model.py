from sklearn.cluster import MiniBatchKMeans
import numpy as np

class Model():
	def __init__(self, input, params, cost_func, DenseCRF):
		self.input = input
		self.cost_func = cost_func
		self.params = params
		self.dense_crf = DenseCRF

	def solve(self):
		self.initialize()
		for i in range(self.params.n_iters):
			print(i+1)
			self.optimize_reflectance()
		return self.get_r_s()

	def initialize(self):
		img_irg = self.input.img_irg
		mask_nz = self.input.mask_nz
		rnd_state = None
		if self.params.fixed_seed:
			rnd_state = np.random.RandomState(seed=59173)
		samples = img_irg[mask_nz[0], mask_nz[1], :]

		# Handling large images
		if samples.shape[0] > self.params.kmeans_max_samples:
			samples = sklearn.utils.shuffle(samples)[:self.params.kmeans_max_samples, :]  

		samples[:, 0] *= self.params.kmeans_intensity_scale
		kmeans = MiniBatchKMeans(n_clusters=self.params.kmeans_n_clusters,
								compute_labels=False, random_state=rnd_state)
		kmeans.fit(samples)
		self.intensities = kmeans.cluster_centers_[:, 0] / self.params.kmeans_intensity_scale
		self.chromaticities = kmeans.cluster_centers_[:, 1:3]

	def optimize_reflectance(self):
		nlabels = self.intensities.shape[0]
		npixels = self.input.mask_nz[0].size
		dcrf = self.dense_crf(npixels, nlabels)
		u_cost = self.cost_func.compute_unary_costs(self.intensities, self.chromaticities)
		dcrf.set_unary_energy(u_cost)
		p_cost = self.cost_func.compute_pairwise_costs(self.intensities, self.chromaticities, self.get_reflectances_rgb())
		p_cost = (self.params.pairwise_weight * p_cost).astype(np.float32)
		dcrf.add_pairwise_energy(pairwise_costs=p_cost, features=self.cost_func.features.copy())
		self.labels_nz = dcrf.map(self.params.n_crf_iters)

	def get_r_s(self):
		s_nz = self.input.image_gray_nz / self.intensities[self.labels_nz]
		r_nz = self.input.image_rgb_nz / np.clip(s_nz, 1e-4, 1e5)[:, np.newaxis]
		r = np.zeros((self.input.rows, self.input.cols, 3), dtype=r_nz.dtype)
		s = np.zeros((self.input.rows, self.input.cols), dtype=s_nz.dtype)
		r[self.input.mask_nz] = r_nz
		s[self.input.mask_nz] = s_nz
		return r, s

	def get_reflectances_rgb(self):
		nlabels = self.intensities.shape[0]
		rgb = np.zeros((nlabels, 3))
		s = 3.0 * self.intensities
		r = self.chromaticities[:, 0]
		g = self.chromaticities[:, 1]
		b = 1.0 - r - g
		rgb[:, 0] = s * r
		rgb[:, 1] = s * g
		rgb[:, 2] = s * b
		return rgb