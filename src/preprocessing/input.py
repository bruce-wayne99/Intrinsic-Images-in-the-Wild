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