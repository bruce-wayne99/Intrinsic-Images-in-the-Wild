import math
import numpy as np
from scipy.misc import imread, imsave

class Input():

	def __init__(self, filename, sRGB=True):

		# load image from filename
		image = self.load_img(filename, sRGB)

		# if given grayscale image
		if image.ndim == 2:
			self.img = np.zeros((image.shape[0], image.shape[1], 3))
			self.img[:, :, :] = image[:, :, np.newaxis]
		else:
			self.img = image.copy()

		# to avoid log(0) ambiguity
		self.img[self.img < 1e-4] = 1e-4
		self.mask = np.ones((self.rows, self.cols), dtype=bool)
		self.mask_nz = np.nonzero(self.mask)
		self.img_irg = self.get_irg()
		self.diag = self.get_diag()

	@property
	def rows(self):
		return self.img.shape[0]

	@property
	def cols(self):
		return self.img.shape[1]

	def load_img(self, filename, sRGB=True):
		img = imread(filename).astype(np.float) / 255.0
		if sRGB:
			return self.srgb_to_rgb(img)
		else:
			return img

	def srgb_to_rgb(self, img):
		ret = np.zeros_like(img)
		idx0 = img <= 0.04045
		idx1 = img > 0.04045
		ret[idx0] = img[idx0] / 12.92
		ret[idx1] = np.power((img[idx1] + 0.055) / 1.055, 2.4)
		return ret

	def get_irg(self):
		irg = np.zeros_like(self.img)
		s = np.sum(self.img, axis=-1)
		irg[..., 0] = s / 3.0
		irg[..., 1] = self.img[..., 0] / s
		irg[..., 2] = self.img[..., 1] / s
		return irg

	def get_diag(self):
		return math.sqrt(np.sum([
				(np.max(nz) - np.min(nz)) ** 2
				for nz in self.mask_nz
			]))

	@property
	def image_gray(self):
		if not hasattr(self, 'img_gray'):
			self.img_gray = np.mean(self.img, axis=2)
		return self.img_gray

	@property
	def image_gray_nz(self):
		return self.image_gray[self.mask_nz]

	@property
	def image_rgb_nz(self):
		return self.img[self.mask_nz]

	@property
	def mask_nnz(self):
		return self.mask_nz[0].size
	