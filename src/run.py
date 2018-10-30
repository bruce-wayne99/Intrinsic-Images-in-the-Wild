import os
import sys
import argparse
from preprocessing.input import Input
from preprocessing.params import HyperParameters
from intrinsic.model import Model
from costs.costs import CostFunction
from krahenbuhl2013.krahenbuhl2013 import DenseCRF

parser = argparse.ArgumentParser(
	description=(
		'Intrinsic Image Decomposition Algorithm\n'
	)
)

parser.add_argument('img', metavar='<file>', help='Image for decomposition')

# exit if input filename is not given
if len(sys.argv) <= 1:
	parser.print_help()
	sys.exit(1)

# extract input image name
args = parser.parse_args()
img_name = args.img
base = os.path.splitext(img_name)[0]
rimg_name = base + '-r.png'
simg_name = base + '-s.png'

# printing input and output formats
print('Input: ' + img_name)
print('Output reflectance image: ' + rimg_name)
print('Output shaded image: ' + simg_name)

# extract input by preprocessing and intialize hyperparameters
input = Input(filename=img_name, sRGB=True)
params = HyperParameters()

# intialize cost function
cost_func = CostFunction(input, params)

# optimize using model
model = Model(input, params, cost_func, DenseCRF)
r_layer, s_layer = model.solve()

# print(r_layer)
# print(s_layer)
# save output
input.save(rimg_name, r_layer, rescale=True, sRGB=True)
input.save(simg_name, s_layer, rescale=True, sRGB=True)