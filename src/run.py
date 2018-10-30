import os
import sys
import argparse
from preprocessing.input import Input

parser = argparse.ArgumentParser(
	description=(
		'Intrinsic Image Algorithm\n'
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

# create the input by preprocessing and intialize hyperparameters
input = Input(filename=img_name, sRGB=True)
# params = HyperParams()