"""
Cell-Pose
Base Configurations class.

Adapted from:
Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

Edited for Cell Segmentation Dataset by Christopher Z. Eddy
contact: eddych@oregonstate.edu

"""

import numpy as np
import os
import datetime


# Base Configuration Class

class Config(object):
	"""Base configuration class. For custom configurations, create a
	sub-class that inherits from this one and override properties
	that need to be changed.
	"""
	# Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
	# Useful if your code needs to do things differently depending on which
	# experiment is running.
	NAME = "CellPose"  # Override in sub-classes

	# NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
	GPU_COUNT = 1

	# Number of images to train with on each GPU. A 12GB GPU can typically
	# handle 2 images of 1024x1024px.
	# Adjust based on your GPU memory and image sizes. Use the highest
	# number that your GPU can handle for best performance.
	IMAGES_PER_GPU = 8

	#Number of training epochs
	EPOCHS = 50

	# Number of training steps per epoch
	# This doesn't need to match the size of the training set. Tensorboard
	# updates are saved at the end of each epoch, so setting this to a
	# smaller number means getting more frequent TensorBoard updates.
	# Validation stats are also calculated at each epoch end and they
	# might take a while, so don't set this too small to avoid spending
	# a lot of time on validation stats.
	STEPS_PER_EPOCH = 50 #was 50

	# Number of validation steps to run at the end of every training epoch.
	# A bigger number improves accuracy of validation stats, but slows
	# down the training.
	VALIDATION_STEPS = 100//IMAGES_PER_GPU#10 # there are 100 examples in validation set. #doesn't matter, eliminated for reproducibility CE 11/24/21
	#could not eliminate for some reason, fit requires to specify validation_steps.
	#still, this argument is eliminated in line 1027 of model.py. Specifies now as the size of the dataset // images_per_gpu

	# Number of classification classes (including background)
	NUM_CLASSES = 1  # Override in sub-classes

	# Input image resizing
	# Generally, use the "square" resizing mode for training and predicting
	# and it should work well in most cases. In this mode, images are scaled
	# up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
	# scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
	# padded with zeros to make it a square so multiple images can be put
	# in one batch.
	# Available resizing modes:
	# none:   No resizing or padding. Return the image unchanged.
	# square: Resize and pad with zeros to get a square image
	#         of size [max_dim, max_dim].
	# pad64:  Pads width and height with zeros to make them multiples of 64.
	#         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
	#         up before padding. IMAGE_MAX_DIM is ignored in this mode.
	#         The multiple of 64 is needed to ensure smooth scaling of feature
	#         maps up and down the 6 levels of the FPN pyramid (2**6=64).
	# crop:   Picks random crops from the image. First, scales the image based
	#         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
	#         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
	#         IMAGE_MAX_DIM is not used in this mode.
	# IMAGE_RESIZE_MODE = "square"
	# IMAGE_MIN_DIM = 800
	# IMAGE_MAX_DIM = 1024
	IMAGE_RESIZE_MODE = "centroid"#"crop"
	IMAGE_MIN_DIM = 256
	IMAGE_MAX_DIM = 256

	# Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
	# up scaling. For example, if set to 2 then images are scaled up to double
	# the width and height, or more, even if MIN_IMAGE_DIM doesn't require it.
	# However, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
	IMAGE_MIN_SCALE = 0

	# Number of color channels per image. RGB = 3, grayscale = 1, RGB-D = 4
	# Changing this requires other changes in the code. See the WIKI for more
	# details: https://github.com/matterport/Mask_RCNN/wiki
	# Currently not actually used in code, although added functionality could be great.
	# Input is expected to be a grayscale stack of images if 3D, otherwise it can be
	# a grayscale or RGB input for 2D.
	IMAGE_CHANNEL_COUNT = 3 #final channel count

	# Learning rate and momentum
	# The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
	# weights to explode. Likely due to differences in optimizer
	# implementation.
	# In CellPose documentation, these parameters follow
	LEARNING_RATE = 0.02
	LEARNING_MOMENTUM = 0.9

	# Weight decay regularization
	WEIGHT_DECAY = 0.00001

	# Loss weights for more precise optimization.
	# Can be used for R-CNN training setup.
	LOSS_WEIGHTS = {
		"CE_Loss_y2": 2.,
		"MSE_Loss_y0": 2.,
		"MSE_Loss_y1": 2.,
		"Dice_Loss_y3": 2.
	}
	#DL was 2., CE was 1. CE 011922
	#was Dice_Loss_y3 and CE_Loss_y1


	AVG_PIX = None #specify if you need dataset average. This will subtract every image
	#by the same amount. If not, it will subtract the average of every image on a
	#per-image basis for training, validation, and inference. Range between 0-255.

	#Augmentation options! Set flags to True or False for augmentations during training.
	#"dense": with probability 0.5, padded image will be z-flipped, flipped left-right with prob 0.5,
	#and then rotated N times with probability 0.5; only if "stack" flag is set
	#to False. Makes the image more object dense and encourages overlapping objects.
	#"stack": if the image slices number less than half INPUT_Z variable, then
	#the image and masks are copied, image and masks are rotated with probability p,
	#and then stacked on top of the self-copy BEFORE padding. Encourages overlapping objects!
	#Setting "dense" or "stack" to True causes a ~20s per iteration increase in
	#training time, mostly due to the increased number of ground truth masks.
	#"noise" really good for overcoming gain setting issues
	#"brightness"
	#"blur_out" good for teaching that very blurry objects are not objects.
	Augmentors = {
		"XYflip": True,
		"Zflip": True,
		"dense": True,
		"stack": True,
		"zoom": False,
		"rotate": True,
		"shear": False,
		"blur": False,
		"blur_out": False,
		"brightness": False,
		"blend": False,
		"noise": False,
		"contrast": False
	}

	#PADDING OPTIONS! Specifies for default training, inference, and PEN visualization.
	# These arguments specify how you want the padding to be applied to the image
	# in the axial direction in load_image_gt and load_image_inference.
	Padding_Opts = {
		"center": True,
		"random": False
	}

	#Projection Enhancement Network (PEN) options! "collect" is how the wide inception inspired
	#convolution branches are put back together to a 3-channel output image for entry into
	#CellPose network (options are "conv", "mean", or "max"). Kernels represent the varying
	#width of the PEN network; each kernel is a size for a convolution on the original image.
	#"block_pool" is how in each branch of the network the convolution should be pooled; options
	#are "conv", "max", and "conv_split". "block_filters" are the number of learned filters in
	#each branch of PEN in the first 3D convolution layer. For a 12gB GPU, with CellPose,
	#block_filters can only be as much as 3.
	PEN_opts = {
		"collect": "conv",
		"kernels": [1,3,5,7,11],
		"block_pool": "conv",
		"block_filters": 3
	}

	#Ground Truth training schema: How to group cells into output channels!
	#Default behavior is to take all z locations of cells in
	#full training image and run k-means to determine which channel they go into for gt.
	#Alternatively, we could run PCA on the x-y-z locations of cells and use one of the
	#principal components to assign the cells to different channels.
	#(1) use k-means only on z-axis locations of cells.
	#(2) use k-means on PCA-projected 3rd component data.
	#(3) Take individual image stack and slice into equal sections. Deprecated
	#(4) Take the full padded image stack and slice into equal sections. Deprecated.
	#(5) Randomly assing cells to a an output channel

	GT_ASSIGN = {'z_kmeans': True,
				 'pca_kmeans': False,
				 'slice_cells': False,
				 'slice_stack': False,
				 'random': False}

	# Train or freeze batch normalization layers
	#     None: Train BN layers. This is the normal mode
	#     False: Freeze BN layers. Good when using a small batch size
	#     True: Set layer in training mode even when predicting
	TRAIN_BN = False  # Defaulting to False since batch size is often small

	# Gradient norm clipping
	GRADIENT_CLIP_NORM = 5.0

	# Specify kernel size to use. Can be list of lists. See Model code for downpass and uppass
	# must be odd integer
	KERNEL_SIZE = 3

	# Specify how many features are to be learned in each layer.
	# length of list must match UNET_DEPTH
	#FROM CellPose
	# The initial convolution (else) is performed in CellPose in order to "mix"
	# the signal from the stained nuclei and the signal from the fluorescence marked cell.
	NUM_LAYER_FEATURES = [32, 64, 128, 256]
	#the point of this is to "mix" the color/input channels before entering UNET
	#if we use head algorithm, then we do not want to use this.

	# Determine how deep for the U-Net structure to go
	# default is 4 blocks down, typical UNET structure
	# Must be integer
	UNET_DEPTH = len(NUM_LAYER_FEATURES)

	#Number of output units
	NUM_OUT = 4

	#Number of channels for Mask, gradients, and borders.
	"""
	OUT_CHANNELS = 1 will run CellPose according to the typical 2D output of CellPose.
	"""
	OUT_CHANNELS = 1

	def __init__(self):
		"""Set values of computed attributes."""
		# Effective batch size
		self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

		#this must be set to three for a proper output of PEN
		if self.PEN_opts["block_pool"]=="max" and self.PEN_opts["collect"]!="conv":
			self.PEN_opts["block_filters"]=3

		if not hasattr(self,"INPUT_IMAGE_SHAPE"):
			# Input image size
			if self.IMAGE_RESIZE_MODE == "crop":
				self.INPUT_IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM,
					self.IMAGE_CHANNEL_COUNT])
			else:
				self.INPUT_IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM,
					self.IMAGE_CHANNEL_COUNT])

	def display(self):
		"""Display Configuration values."""
		print("\nConfigurations:")
		for a in dir(self):
			if not a.startswith("__") and not callable(getattr(self, a)):
				print("{:30} {}".format(a, getattr(self, a)))
		print("\n")

	def write_config_txt(self,fpath):
		""" Write configuration values to .txt file """
		with open(os.path.join(fpath,'README.txt'), 'w') as f:
			f.write("README \n")
			f.write("Training begin date and time: {:%Y%m%dT%H%M%S}".format(datetime.datetime.now()))
			f.write("\n")
			f.write("Notes:\n")
			f.write("CONFIGURATION SETTINGS: \n")
			for a in dir(self):
				if not a.startswith("__") and not callable(getattr(self, a)):
					f.write("{:30} {}".format(a, getattr(self, a)))
					f.write("\n")

class Cell3DConfig(Config):
	INPUT_DIM = "3D"
	assert INPUT_DIM in ["2D", "3D"]
	INPUT_Z = 27
	IMAGE_CHANNEL_COUNT = 1
	IMAGE_RESIZE_MODE = "centroid"
	IMAGE_MIN_DIM = 256
	IMAGE_MAX_DIM = 256
	"""
	OUT_CHANNELS
	Needs to change if you are using Cell3DConfig. Inference should be set
	equivalent to the training procedure. Should be set depending on training
	dataset.
	"""
	OUT_CHANNELS = 3

	# if np.logical_or(Config.Augmentors['zoom'], Config.Augmentors['shear']):
	# 	IMAGE_RESIZE_MODE = "crop"
	# Input image size
	if IMAGE_RESIZE_MODE == "crop":
		INPUT_IMAGE_SHAPE = np.array([IMAGE_MIN_DIM, IMAGE_MIN_DIM, INPUT_Z,
			IMAGE_CHANNEL_COUNT])
	else:
		INPUT_IMAGE_SHAPE = np.array([IMAGE_MAX_DIM, IMAGE_MAX_DIM, INPUT_Z,
			IMAGE_CHANNEL_COUNT])


class Cell2DConfig(Config):
	INPUT_DIM = "2D"
	assert INPUT_DIM in ["2D", "3D"]
	INPUT_Z = 27
	"""
	OUT_CHANNELS
	Needs to change if you are using Cell3DConfig. Inference should be set
	equivalent to the training procedure. Should be set depending on training
	dataset.
	"""
	OUT_CHANNELS = 1

	#options here are "linear" or "max". If max, this will do a max projection to a gray scale input
	#essentially input channels will all be the same.
	#if linear, then we do a linear color assingnment using RGB to the three color channels
	#based on z-position. See utils.py.
	project_mode = "linear"

	#options here are True or False. If True, we pad images to have Z (INPUT_Z)
	#slices BEFORE depth projection. The reason you may wish to do this is that
	#you may want the projected color shade difference to be reflective of a
	#constant deltaZ. If this is set to False, each image is uniquely stretched
	#based on its own number of Z slices, and therefore there is not a constant
	#learned deltaZ.
	if project_mode=="linear":
		pad_before_linear = False

	if np.logical_or(Config.Augmentors['zoom'], Config.Augmentors['shear']):
		IMAGE_RESIZE_MODE = "crop"

	assert project_mode in ["max", "linear"], "Cell2DConfig requires 'project_mode' to be either 'max' or 'linear'."
