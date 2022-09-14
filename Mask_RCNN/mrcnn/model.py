"""
Mask R-CNN
The main Mask R-CNN model implementation.
Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import random
import datetime
import re
import math
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.keras.layers as KE
from tensorflow.python.eager import context
import tensorflow.keras.models as KM
import copy

from mrcnn import utils

# Requires TensorFlow 2.0+
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("2.0")

tf.compat.v1.disable_eager_execution()

############################################################
#  Utility Functions
############################################################


class AnchorsLayer(KL.Layer):
	def __init__(self, anchors, name="anchors", **kwargs):
		super(AnchorsLayer, self).__init__(name=name, **kwargs)
		self.anchors = tf.Variable(anchors)

	def call(self, dummy):
		return self.anchors

	def get_config(self):
		config = super(AnchorsLayer, self).get_config()
		return config

# class AnchorsLayer(tf.keras.layers.Layer):
#         def __init__(self, name="anchors", **kwargs):
#             super(AnchorsLayer, self).__init__(name=name, **kwargs)
#
#         def call(self, anchor):
#             return anchor
#
#         def get_config(self) :
#             config = super(AnchorsLayer, self).get_config()
#             return config


def log(text, array=None):
	"""Prints a text message. And, optionally, if a Numpy array is provided it
	prints it's shape, min, and max values.
	"""
	if array is not None:
		text = text.ljust(25)
		text += ("shape: {:20}  ".format(str(array.shape)))
		if array.size:
			text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(),array.max()))
		else:
			text += ("min: {:10}  max: {:10}".format("",""))
		text += "  {}".format(array.dtype)
	print(text)


class BatchNorm(KL.BatchNormalization):
	"""Extends the Keras BatchNormalization class to allow a central place
	to make changes if needed.
	Batch normalization has a negative effect on training if batches are small
	so this layer is often frozen (via setting in Config class) and functions
	as linear layer.
	"""
	def call(self, inputs, training=None):
		"""
		Note about training values:
			None: Train BN layers. This is the normal mode
			False: Freeze BN layers. Good when batch size is small
			True: (don't use). Set layer in training mode even when making inferences
		"""
		return super(self.__class__, self).call(inputs, training=training)


def compute_backbone_shapes(config, image_shape):
	"""Computes the width and height of each stage of the backbone network.
	Returns:
		[N, (height, width)]. Where N is the number of stages
	"""
	if callable(config.BACKBONE):
		return config.COMPUTE_BACKBONE_SHAPE(image_shape)

	# Currently supports ResNet only
	assert config.BACKBONE in ["resnet50", "resnet101"]
	return np.array(
		[[int(math.ceil(image_shape[0] / stride)),
			int(math.ceil(image_shape[1] / stride))]
			for stride in config.BACKBONE_STRIDES])


############################################################
#  Resnet Graph
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

def identity_block(input_tensor, kernel_size, filters, stage, block,
				   use_bias=True, train_bn=True):
	"""The identity_block is the block that has no conv layer at shortcut
	# Arguments
		input_tensor: input tensor
		kernel_size: default 3, the kernel size of middle conv layer at main path
		filters: list of integers, the nb_filters of 3 conv layer at main path
		stage: integer, current stage label, used for generating layer names
		block: 'a','b'..., current block label, used for generating layer names
		use_bias: Boolean. To use or not use a bias in conv layers.
		train_bn: Boolean. Train or freeze Batch Norm layers
	"""
	nb_filter1, nb_filter2, nb_filter3 = filters
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
				  use_bias=use_bias)(input_tensor)
	x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
	x = KL.Activation('relu')(x)

	x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
				  name=conv_name_base + '2b', use_bias=use_bias)(x)
	x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
	x = KL.Activation('relu')(x)

	x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
				  use_bias=use_bias)(x)
	x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

	x = KL.Add()([x, input_tensor])
	x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
	return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
			   strides=(2, 2), use_bias=True, train_bn=True):
	"""conv_block is the block that has a conv layer at shortcut
	# Arguments
		input_tensor: input tensor
		kernel_size: default 3, the kernel size of middle conv layer at main path
		filters: list of integers, the nb_filters of 3 conv layer at main path
		stage: integer, current stage label, used for generating layer names
		block: 'a','b'..., current block label, used for generating layer names
		use_bias: Boolean. To use or not use a bias in conv layers.
		train_bn: Boolean. Train or freeze Batch Norm layers
	Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
	And the shortcut should have subsample=(2,2) as well
	"""
	nb_filter1, nb_filter2, nb_filter3 = filters
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
				  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
	x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
	x = KL.Activation('relu')(x)

	x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
				  name=conv_name_base + '2b', use_bias=use_bias)(x)
	x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
	x = KL.Activation('relu')(x)

	x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
				  '2c', use_bias=use_bias)(x)
	x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

	shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
						 name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
	shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)

	x = KL.Add()([x, shortcut])
	x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
	return x


def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
	"""Build a ResNet graph.
		architecture: Can be resnet50 or resnet101
		stage5: Boolean. If False, stage5 of the network is not created
		train_bn: Boolean. Train or freeze Batch Norm layers
	"""
	assert architecture in ["resnet50", "resnet101"]
	# Stage 1
	x = KL.ZeroPadding2D((3, 3))(input_image)
	x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
	x = BatchNorm(name='bn_conv1')(x, training=train_bn)
	x = KL.Activation('relu')(x)
	C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
	# Stage 2
	x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
	x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
	C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
	# Stage 3
	x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
	C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
	# Stage 4
	x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
	block_count = {"resnet50": 5, "resnet101": 22}[architecture]
	for i in range(block_count):
		x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
	C4 = x
	# Stage 5
	if stage5:
		x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
		x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
		C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
	else:
		C5 = None
	return [C1, C2, C3, C4, C5]

#############################################################################
##################################STEM ZOO###################################
#############################################################################
"""
There are two ways to tackle the stem issue:
(1) You can use 3D convolutions and essentially split the Z stack into 3 z-images
depending on the size of the convolution kernel. Then, move these 3 images to the
channels portion and squeeze. In this way, z information is stored throughout the
network. Ideally, each learned filter in RGB should be the same.
(2) You can use 3D convolutions and learn 3 useful filters that is then pooled down
to a 2D image. In this way 3D information is ultimately used, but lost after stem.
"""

"""
From CellPose
"""
####################(1)STEM GRAPHS WITH 3D INFO IN CHANNELS#####################
def stem_block_z(input_tensor, input_z, k_xy, k_z, train_bn=True):
	"""
	INPUTS:
	input_tensor = tensor to apply stem block on. Should be  shape = B x H x W x Z x 1
	input_z = config.INPUT_DIM. Needed to calculate optimal strides.
	k_xy = kernel size in x and y dimensions
	k_z = kernel size in z dimension.
	OUTPUTS:
	x = tensor with shape = B x H x W x 3
	where 3 corresponds to reduced z images.
	"""
	assert input_z % 2 != 0 #input_z must be odd.
	assert k_z % 2 != 0 #kernel must also be odd valued.

	conv_name = 'stem_conv_kernel_z' + str(k_z) + '_xy' + str(k_xy)
	bn_name = 'stem_bn_kernel_z' + str(k_z) + '_xy' + str(k_xy)
	stem_squeeze_name = 'stem_squeeze_kernel_z' + str(k_z) + '_xy' + str(k_xy)
	act_name = 'stem_act_kernel_z' + str(k_z) + '_xy' + str(k_xy)
	pool_name = 'stem_pool_kernel_z' + str(k_z) + '_xy' + str(k_xy)

	if input_z/k_z > 3:
		#in this case, we will have a loss of information with the current kernel.
		#to counter this, we use convolution followed by max pooling.
		#convolution with kernel at maximum stride w/o loss of info.
		#determine z stride
		s = max([s+1 for s in range(k_z) if int((input_z - k_z)/(s+1)) == ((input_z - k_z)/(s+1))])
		#do proper zero padding first.
		pad_size = int(np.floor(k_xy/2))#k_xy//2 #does this round up or down?? Displays different behaviors. Should be a floor divide.
		x = KL.ZeroPadding3D((pad_size,pad_size,0))(input_tensor)
		x = KL.Conv3D(1, (k_xy, k_xy, k_z), strides=(1,1,s), name = conv_name)(x)
		x = KL.BatchNormalization(trainable=train_bn, name = bn_name)(x)
		#x = KL.BatchNormalization()(x)
		x = KL.Activation('relu', name = act_name)(x)
		#x = KL.Activation('relu')(x)
		#pooling
		#determine how many "z" images there are after the previous convolution.
		c = int(((input_z - k_z)/s) + 1)
		#determine z pooling size to reduce to 3 images
		P = int(min([p+1 for p in range(c) if (c/(p+1))<=3 and int((c-(p+1))/2)==((c-(p+1))/2)]))
		#first requirement is that in can be settled to 3 images.
		#second requirement is that the corresponding stride length can be an integer.
		#determine z pooling stride to reduce to 3 images.
		s = int((c-P) / 2)
		x = KL.MaxPooling3D(pool_size=(1,1,P), strides=(1,1,s), name = pool_name)(x)
	else:
		#in this case, 3 z-images can be obtained based on the size of the kernel.
		#determine optimal stride size.
		s = int((input_z - k_z) / 2)
		pad_size = int(np.floor(k_xy/2))#k_xy//2 #does this round up or down?? Displays different behaviors. Should be a floor divide.
		x = KL.ZeroPadding3D((pad_size,pad_size,0))(input_tensor)
		x = KL.Conv3D(1, (k_xy, k_xy, k_z), strides=(1,1,s), name = conv_name)(x)
		x = KL.BatchNormalization(trainable=train_bn, name = bn_name)(x)
		#x = KL.BatchNormalization()(x)
		x = KL.Activation('relu', name = act_name)(x)
		#x = KL.Activation('relu')(x)
	#squeeze layer
	x = KL.Lambda(lambda x: K.squeeze(x,-1),name=stem_squeeze_name)(x)
	return x


def stem_graph_max_z(input_image, input_z, train_bn=True):
	"""
	Turn 3D input to 2D input for ResNet graph.
	Reduce the Z channels to a 3 channel output.
	the plan is to do multiscale features and a max projection in each.
	Then to accumulate them with an average pooling layer.

	Well, the input is actually 2D.
	input shape is [Batch, H, W, Z, 1] where 1 is channel=gray
	output shape is [Batch, H, W, 3]
	"""
	#input has already been padded in z as necessary.
	#start with a large convolution to pick up the large spatial features in the image?
	#resnet begins with a 7x7 conv, 64 filters, followed by a pooling layer.
	#this assures a large-receptive field with strong contextual information in the first layer.
	#with 64 filters to assure a rich set of primary features.
	#pad edges of x,y so we do not lose spatial information.
	#note, these are not the same as a max projection because weights are learned along the way.
	x_1 = stem_block_z(input_image, input_z=input_z, k_xy=1, k_z=1,
				   train_bn=train_bn)#(input_image)
	x_3 = stem_block_z(input_image, input_z=input_z, k_xy=3, k_z=3,
				   train_bn=train_bn)#(input_image)
	x_5 = stem_block_z(input_image, input_z=input_z, k_xy=5, k_z=5,
				   train_bn=train_bn)#(input_image)
	x_7 = stem_block_z(input_image, input_z=input_z, k_xy=7, k_z=7,
				   train_bn=train_bn)#(input_image)
	x_11 = stem_block_z(input_image, input_z=input_z, k_xy=11, k_z=11,
				   train_bn=train_bn)#(input_image)

	#so now each output should be an bn x X x Y x 3
	#at this point, we want to do an average pooling.
	x = KL.Average()([x_1,x_3,x_5,x_7,x_11])

	#this may help with training to make input into resnet normal
	#x = KL.Lambda(lambda x: x - tf.reduce_min(tf.reshape(x,[-1,1]),axis=0) ,name="stem_rescale1")(x)
	x = KL.Lambda(lambda x: x / tf.reduce_max(tf.reshape(x,[-1,1]),axis=0) ,name="stem_rescale")(x)


	return x

####################(2)STEM GRAPHS WITH 3D INFO IN FEATURES#####################
def stem_block(input_tensor, kernel_size, num_filters,
			   use_bias=True, train_bn=True, pool_mode="max"):
	"""
	For this, we need to pass the kernel sizes, that should be it.
	"""
	conv_name = 'stem_conv_kernel' + str(kernel_size)
	bn_name = 'stem_bn_kernel' + str(kernel_size)
	stem_squeeze_name = 'stem_squeeze_kernel' + str(kernel_size)
	act_name = 'stem_act_kernel' + str(kernel_size)
	pool_name = 'stem_pool' + str(kernel_size)
	#some stuff
	t_shape = input_tensor.get_shape().as_list()
	#asser that the kernel size used must be less than the dimensions of the image
	assert any(x>=kernel_size for x in t_shape if x is not None)
	pad_size = kernel_size//2 #floor divide
	x = KL.ZeroPadding3D((pad_size,pad_size,0))(input_tensor)
	x = KL.Conv3D(num_filters if pool_mode=="conv" else 3, (kernel_size,kernel_size,kernel_size), name = conv_name)(x) #name=conv_name_base + '2a'
	x = KL.BatchNormalization(trainable=train_bn, name = bn_name)(x)
	x = KL.Activation('relu', name = act_name)(x)
	p_size = max(0, kernel_size - 1)
	#this is a question if batch size counts for t_shape...
	p_size = t_shape[3] - p_size
	#p size is the pooling size of the z dimension.

	if pool_mode == "max":
		#this does maximum projection
		x = KL.MaxPooling3D(pool_size=(1, 1, p_size), strides=(1,1,1), name = pool_name+"_max")(x)
	#CANNOT DO A CONVOLUTION AS SIZE CHANGES BASED ON p_size.
	#can we predict for sure, how large p_size is? well, tensor z is from config.INPUT+
	elif pool_mode == "conv":
		#could do a conv instead.
		x = KL.Conv3D(3, (1,1,p_size), name = pool_name+"_conv")(x)
		x = KL.BatchNormalization(trainable=train_bn, name = pool_name+"_bn")(x)
		x = KL.Activation('relu', name = pool_name+"_act")(x)
	else:
		print("Pooling option not available...")
		import pdb;pdb.set_trace()


	#squeeze the z dimension, which is now down to one.
	x = KL.Lambda(lambda x: K.squeeze(x,-2),name=stem_squeeze_name)(x)

	return x

def stem_graph_max(input_image, batch_size, z_slices, PEN_opts, train_bn=True):
	"""
	Turn 3D input to 2D input for ResNet graph.
	Reduce the Z channels to a 3 channel output.
	the plan is to do multiscale features and a max projection in each.
	Then to accumulate them with an average pooling layer.

	Well, the input is actually 2D.
	input shape is [Batch, H, W, Z, 1] where 1 is channel=gray
	output shape is [Batch, H, W, 3]
	"""
	#input has already been padded in z as necessary.
	#start with a large convolution to pick up the large spatial features in the image?
	#resnet begins with a 7x7 conv, 64 filters, followed by a pooling layer.
	#this assures a large-receptive field with strong contextual information in the first layer.
	#with 64 filters to assure a rich set of primary features.
	#pad edges of x,y so we do not lose spatial information.
	#note, these are not the same as a max projection because weights are learned along the way.

	kernels = PEN_opts['kernels']

	x = [stem_block(input_image, kernel_size=k, num_filters=PEN_opts['block_filters'], use_bias = True, train_bn=train_bn, pool_mode=PEN_opts['block_pool']) for k in kernels]

	#so now each output should be an bn x X x Y x 3
	#at this point, we want to do an average pooling.
	if PEN_opts['collect'] == "mean":
		x = KL.Average()(x) #OLD CE 02042022
	elif PEN_opts['collect'] == "max":
		x = KL.Maximum()(x) #Need to check if this output is correct.
	elif PEN_opts['collect'] == "conv":
		#better, stack them in the z dimension and then run a 3D convolution.
		x = KL.Lambda(lambda x: tf.stack(x,axis=-2),name="stem_stack_collect")(x)
		x = KL.Conv3D(3, (1,1,len(kernels)), name = "stem_conv_collect")(x) #name=conv_name_base + '2a'
		x = KL.BatchNormalization(trainable=train_bn, name = "stem_bn_collect")(x)
		x = KL.Activation('relu', name = "stem_act_collect")(x)
		#squeeze the z dimension, which is now down to one.
		x = KL.Lambda(lambda x: K.squeeze(x,-2),name="stem_collect_squeeze")(x)
		#check shape, should be B x H x W x 3
	else:
		print("The designated config.PEN_opts['collect'] flag is not an option.")
		import pdb;pdb.set_trace()

	#this may help with training to make input into resnet normal
	#x = KL.Lambda(lambda x: x - tf.reduce_min(tf.reshape(x,[-1,1]),axis=0) ,name="stem_rescale_min")(x)
	x = KL.Lambda(lambda x: x - tf.reshape(tf.reduce_min(tf.reshape(x, [batch_size,-1]),axis=1),(batch_size,)+(1,)*3), name="stem_rescale_min")(x)
	x = KL.Lambda(lambda x: x / tf.reshape(tf.reduce_max(tf.reshape(x, [batch_size,-1]),axis=1),(batch_size,)+(1,)*3), name="stem_normalize")(x)
	#x = KL.Lambda(lambda x: x / tf.reduce_max(tf.reshape(tf.abs(x),[-1,1]),axis=0) ,name="stem_normalize")(x)
	return x

def stem_project_test(input_image, train_bn=True):
	"""
	Turn the 3D input into a max projection, the same method used in the 2D input
	"""
	#input has already been padded in z as necessary.
	#start with a large convolution to pick up the large spatial features in the image?
	#resnet begins with a 7x7 conv, 64 filters, followed by a pooling layer.
	#this assures a large-receptive field with strong contextual information in the first layer.
	#with 64 filters to assure a rich set of primary features.
	#pad edges of x,y so we do not lose spatial information.
	t_shape = input_image.get_shape().as_list()
	#for x_1, do a literal max projection.
	x_1 =  KL.MaxPooling3D(pool_size=(1, 1, t_shape[3]), strides=(1,1,1))(input_image)
	#squeeze the z dimension, which is now down to one.
	x_1 = KL.Lambda(lambda x: K.squeeze(x,-2),name='stem_squeeze')(x_1)
	#now go from [X,Y,1] to [X,Y,3] by copying to the other channels.
	x_1 = KL.Concatenate(axis=-1)([x_1,x_1,x_1])
	#stem_block(input_image, kernel_size=1,

	return x_1


#############################################################################

############################################################
#  Proposal Layer
############################################################

def apply_box_deltas_graph(boxes, deltas):
	"""Applies the given deltas to the given boxes.
	boxes: [N, (y1, x1, y2, x2)] boxes to update
	deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
	"""
	# Convert to y, x, h, w
	height = boxes[:, 2] - boxes[:, 0]
	width = boxes[:, 3] - boxes[:, 1]
	center_y = boxes[:, 0] + 0.5 * height
	center_x = boxes[:, 1] + 0.5 * width
	# Apply deltas
	center_y += deltas[:, 0] * height
	center_x += deltas[:, 1] * width
	height *= tf.exp(deltas[:, 2])
	width *= tf.exp(deltas[:, 3])
	# Convert back to y1, x1, y2, x2
	y1 = center_y - 0.5 * height
	x1 = center_x - 0.5 * width
	y2 = y1 + height
	x2 = x1 + width
	result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
	return result


def clip_boxes_graph(boxes, window):
	"""
	boxes: [N, (y1, x1, y2, x2)]
	window: [4] in the form y1, x1, y2, x2
	"""
	# Split
	wy1, wx1, wy2, wx2 = tf.split(window, 4)
	y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
	# Clip
	y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
	x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
	y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
	x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
	clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
	clipped.set_shape((clipped.shape[0], 4))
	return clipped


class ProposalLayer(KE.Layer):
	"""Receives anchor scores and selects a subset to pass as proposals
	to the second stage. Filtering is done based on anchor scores and
	non-max suppression to remove overlaps. It also applies bounding
	box refinement deltas to anchors.
	Inputs:
		rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
		rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
		anchors: [batch, num_anchors, (y1, x1, y2, x2)] anchors in normalized coordinates
	Returns:
		Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
	"""

	def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
		super(ProposalLayer, self).__init__(**kwargs)
		self.config = config
		self.proposal_count = proposal_count
		self.nms_threshold = nms_threshold

	def get_config(self):
		config = super(ProposalLayer, self).get_config()
		config["config"] = self.config.to_dict()
		config["proposal_count"] = self.proposal_count
		config["nms_threshold"] = self.nms_threshold
		return config

	def call(self, inputs):
		# Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
		scores = inputs[0][:, :, 1]
		# Box deltas [batch, num_rois, 4]
		deltas = inputs[1]
		deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
		# Anchors
		anchors = inputs[2]

		# Improve performance by trimming to top anchors by score
		# and doing the rest on the smaller subset.
		pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(input=anchors)[1])
		ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
						 name="top_anchors").indices
		scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y),
								   self.config.IMAGES_PER_GPU)
		deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),
								   self.config.IMAGES_PER_GPU)
		pre_nms_anchors = utils.batch_slice([anchors, ix], lambda a, x: tf.gather(a, x),
									self.config.IMAGES_PER_GPU,
									names=["pre_nms_anchors"])

		# Apply deltas to anchors to get refined anchors.
		# [batch, N, (y1, x1, y2, x2)]
		boxes = utils.batch_slice([pre_nms_anchors, deltas],
								  lambda x, y: apply_box_deltas_graph(x, y),
								  self.config.IMAGES_PER_GPU,
								  names=["refined_anchors"])

		# Clip to image boundaries. Since we're in normalized coordinates,
		# clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
		window = np.array([0, 0, 1, 1], dtype=np.float32)
		boxes = utils.batch_slice(boxes,
								  lambda x: clip_boxes_graph(x, window),
								  self.config.IMAGES_PER_GPU,
								  names=["refined_anchors_clipped"])

		# Filter out small boxes
		# According to Xinlei Chen's paper, this reduces detection accuracy
		# for small objects, so we're skipping it.

		# Non-max suppression
		def nms(boxes, scores):
			indices = tf.image.non_max_suppression(
				boxes, scores, self.proposal_count,
				self.nms_threshold, name="rpn_non_max_suppression")
			proposals = tf.gather(boxes, indices)
			# Pad if needed
			padding = tf.maximum(self.proposal_count - tf.shape(input=proposals)[0], 0)
			proposals = tf.pad(tensor=proposals, paddings=[(0, padding), (0, 0)])
			return proposals
		proposals = utils.batch_slice([boxes, scores], nms,
									  self.config.IMAGES_PER_GPU)

		if not context.executing_eagerly():
			# Infer the static output shape:
			out_shape = self.compute_output_shape(None)
			proposals.set_shape(out_shape)
		return proposals

	def compute_output_shape(self, input_shape):
		return None, self.proposal_count, 4


############################################################
#  ROIAlign Layer
############################################################

def log2_graph(x):
	"""Implementation of Log2. TF doesn't have a native implementation."""
	return tf.math.log(x) / tf.math.log(2.0)


class PyramidROIAlign(KE.Layer):
	"""Implements ROI Pooling on multiple levels of the feature pyramid.
	Params:
	- pool_shape: [pool_height, pool_width] of the output pooled regions. Usually [7, 7]
	Inputs:
	- boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
			 coordinates. Possibly padded with zeros if not enough
			 boxes to fill the array.
	- image_meta: [batch, (meta data)] Image details. See compose_image_meta()
	- feature_maps: List of feature maps from different levels of the pyramid.
					Each is [batch, height, width, channels]
	Output:
	Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].
	The width and height are those specific in the pool_shape in the layer
	constructor.
	"""

	def __init__(self, pool_shape, **kwargs):
		super(PyramidROIAlign, self).__init__(**kwargs)
		self.pool_shape = tuple(pool_shape)

	def get_config(self):
		config = super(PyramidROIAlign, self).get_config()
		config['pool_shape'] = self.pool_shape
		return config

	def call(self, inputs):
		# Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
		boxes = inputs[0]

		# Image meta
		# Holds details about the image. See compose_image_meta()
		image_meta = inputs[1]

		# Feature Maps. List of feature maps from different level of the
		# feature pyramid. Each is [batch, height, width, channels]
		feature_maps = inputs[2:]

		# Assign each ROI to a level in the pyramid based on the ROI area.
		y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
		h = y2 - y1
		w = x2 - x1
		# Use shape of first image. Images in a batch must have the same size.
		image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]
		# Equation 1 in the Feature Pyramid Networks paper. Account for
		# the fact that our coordinates are normalized here.
		# e.g. a 224x224 ROI (in pixels) maps to P4
		image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
		roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
		roi_level = tf.minimum(5, tf.maximum(
			2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
		roi_level = tf.squeeze(roi_level, 2)

		# Loop through levels and apply ROI pooling to each. P2 to P5.
		pooled = []
		box_to_level = []
		for i, level in enumerate(range(2, 6)):
			ix = tf.compat.v1.where(tf.equal(roi_level, level))
			level_boxes = tf.gather_nd(boxes, ix)

			# Box indices for crop_and_resize.
			box_indices = tf.cast(ix[:, 0], tf.int32)

			# Keep track of which box is mapped to which level
			box_to_level.append(ix)

			# Stop gradient propogation to ROI proposals
			level_boxes = tf.stop_gradient(level_boxes)
			box_indices = tf.stop_gradient(box_indices)

			# Crop and Resize
			# From Mask R-CNN paper: "We sample four regular locations, so
			# that we can evaluate either max or average pooling. In fact,
			# interpolating only a single value at each bin center (without
			# pooling) is nearly as effective."
			#
			# Here we use the simplified approach of a single value per bin,
			# which is how it's done in tf.crop_and_resize()
			# Result: [batch * num_boxes, pool_height, pool_width, channels]
			pooled.append(tf.image.crop_and_resize(
				feature_maps[i], level_boxes, box_indices, self.pool_shape,
				method="bilinear"))

		# Pack pooled features into one tensor
		pooled = tf.concat(pooled, axis=0)

		# Pack box_to_level mapping into one array and add another
		# column representing the order of pooled boxes
		box_to_level = tf.concat(box_to_level, axis=0)
		box_range = tf.expand_dims(tf.range(tf.shape(input=box_to_level)[0]), 1)
		box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
								 axis=1)

		# Rearrange pooled features to match the order of the original boxes
		# Sort box_to_level by batch then box index
		# TF doesn't have a way to sort by two columns, so merge them and sort.
		sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
		ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
			input=box_to_level)[0]).indices[::-1]
		ix = tf.gather(box_to_level[:, 2], ix)
		pooled = tf.gather(pooled, ix)

		# Re-add the batch dimension
		shape = tf.concat([tf.shape(input=boxes)[:2], tf.shape(input=pooled)[1:]], axis=0)
		pooled = tf.reshape(pooled, shape)
		return pooled

	def compute_output_shape(self, input_shape):
		return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1], )


############################################################
#  Detection Target Layer
############################################################

def overlaps_graph(boxes1, boxes2):
	"""Computes IoU overlaps between two sets of boxes.
	boxes1, boxes2: [N, (y1, x1, y2, x2)].
	"""
	# 1. Tile boxes2 and repeat boxes1. This allows us to compare
	# every boxes1 against every boxes2 without loops.
	# TF doesn't have an equivalent to np.repeat() so simulate it
	# using tf.tile() and tf.reshape.
	b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
							[1, 1, tf.shape(input=boxes2)[0]]), [-1, 4])
	b2 = tf.tile(boxes2, [tf.shape(input=boxes1)[0], 1])
	# 2. Compute intersections
	b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
	b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
	y1 = tf.maximum(b1_y1, b2_y1)
	x1 = tf.maximum(b1_x1, b2_x1)
	y2 = tf.minimum(b1_y2, b2_y2)
	x2 = tf.minimum(b1_x2, b2_x2)
	intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
	# 3. Compute unions
	b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
	b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
	union = b1_area + b2_area - intersection
	# 4. Compute IoU and reshape to [boxes1, boxes2]
	iou = intersection / union
	overlaps = tf.reshape(iou, [tf.shape(input=boxes1)[0], tf.shape(input=boxes2)[0]])
	return overlaps


def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config):
	"""Generates detection targets for one image. Subsamples proposals and
	generates target class IDs, bounding box deltas, and masks for each.
	Inputs:
	proposals: [POST_NMS_ROIS_TRAINING, (y1, x1, y2, x2)] in normalized coordinates. Might
			   be zero padded if there are not enough proposals.
	gt_class_ids: [MAX_GT_INSTANCES] int class IDs
	gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
	gt_masks: [height, width, MAX_GT_INSTANCES] of boolean type.
	Returns: Target ROIs and corresponding class IDs, bounding box shifts,
	and masks.
	rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
	class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
	deltas: [TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))]
	masks: [TRAIN_ROIS_PER_IMAGE, height, width]. Masks cropped to bbox
		   boundaries and resized to neural network output size.
	Note: Returned arrays might be zero padded if not enough target ROIs.
	"""
	# Assertions
	asserts = [
		tf.Assert(tf.greater(tf.shape(input=proposals)[0], 0), [proposals],
				  name="roi_assertion"),
	]
	with tf.control_dependencies(asserts):
		proposals = tf.identity(proposals)

	# Remove zero padding
	proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
	gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
	gt_class_ids = tf.boolean_mask(tensor=gt_class_ids, mask=non_zeros,
								   name="trim_gt_class_ids")
	gt_masks = tf.gather(gt_masks, tf.compat.v1.where(non_zeros)[:, 0], axis=2,
						 name="trim_gt_masks")

	# Handle COCO crowds
	# A crowd box in COCO is a bounding box around several instances. Exclude
	# them from training. A crowd box is given a negative class ID.
	crowd_ix = tf.compat.v1.where(gt_class_ids < 0)[:, 0]
	non_crowd_ix = tf.compat.v1.where(gt_class_ids > 0)[:, 0]
	crowd_boxes = tf.gather(gt_boxes, crowd_ix)
	gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
	gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
	gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

	# Compute overlaps matrix [proposals, gt_boxes]
	overlaps = overlaps_graph(proposals, gt_boxes)

	# Compute overlaps with crowd boxes [proposals, crowd_boxes]
	crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
	crowd_iou_max = tf.reduce_max(input_tensor=crowd_overlaps, axis=1)
	no_crowd_bool = (crowd_iou_max < 0.001)

	# Determine positive and negative ROIs
	roi_iou_max = tf.reduce_max(input_tensor=overlaps, axis=1)
	# 1. Positive ROIs are those with >= 0.5 IoU with a GT box
	positive_roi_bool = (roi_iou_max >= 0.5)
	positive_indices = tf.compat.v1.where(positive_roi_bool)[:, 0]
	# 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
	negative_indices = tf.compat.v1.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

	# Subsample ROIs. Aim for 33% positive
	# Positive ROIs
	positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
						 config.ROI_POSITIVE_RATIO)
	positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
	positive_count = tf.shape(input=positive_indices)[0]
	# Negative ROIs. Add enough to maintain positive:negative ratio.
	r = 1.0 / config.ROI_POSITIVE_RATIO
	negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
	negative_indices = tf.random.shuffle(negative_indices)[:negative_count]
	# Gather selected ROIs
	positive_rois = tf.gather(proposals, positive_indices)
	negative_rois = tf.gather(proposals, negative_indices)

	# Assign positive ROIs to GT boxes.
	positive_overlaps = tf.gather(overlaps, positive_indices)
	roi_gt_box_assignment = tf.cond(
		pred=tf.greater(tf.shape(input=positive_overlaps)[1], 0),
		true_fn=lambda: tf.argmax(input=positive_overlaps, axis=1),
		false_fn=lambda: tf.cast(tf.constant([]), tf.int64)
	)
	roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
	roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

	# Compute bbox refinement for positive ROIs
	deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
	deltas /= config.BBOX_STD_DEV

	# Assign positive ROIs to GT masks
	# Permute masks to [N, height, width, 1]
	transposed_masks = tf.expand_dims(tf.transpose(a=gt_masks, perm=[2, 0, 1]), -1)
	# Pick the right mask for each ROI
	roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

	# Compute mask targets
	boxes = positive_rois
	if config.USE_MINI_MASK:
		# Transform ROI coordinates from normalized image space
		# to normalized mini-mask space.
		y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
		gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
		gt_h = gt_y2 - gt_y1
		gt_w = gt_x2 - gt_x1
		y1 = (y1 - gt_y1) / gt_h
		x1 = (x1 - gt_x1) / gt_w
		y2 = (y2 - gt_y1) / gt_h
		x2 = (x2 - gt_x1) / gt_w
		boxes = tf.concat([y1, x1, y2, x2], 1)
	box_ids = tf.range(0, tf.shape(input=roi_masks)[0])
	masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
									 box_ids,
									 config.MASK_SHAPE)
	# Remove the extra dimension from masks.
	masks = tf.squeeze(masks, axis=3)

	# Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
	# binary cross entropy loss.
	masks = tf.round(masks)

	# Append negative ROIs and pad bbox deltas and masks that
	# are not used for negative ROIs with zeros.
	rois = tf.concat([positive_rois, negative_rois], axis=0)
	N = tf.shape(input=negative_rois)[0]
	P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(input=rois)[0], 0)
	rois = tf.pad(tensor=rois, paddings=[(0, P), (0, 0)])
	roi_gt_boxes = tf.pad(tensor=roi_gt_boxes, paddings=[(0, N + P), (0, 0)])
	roi_gt_class_ids = tf.pad(tensor=roi_gt_class_ids, paddings=[(0, N + P)])
	deltas = tf.pad(tensor=deltas, paddings=[(0, N + P), (0, 0)])
	masks = tf.pad(tensor=masks, paddings=[[0, N + P], (0, 0), (0, 0)])

	return rois, roi_gt_class_ids, deltas, masks


class DetectionTargetLayer(KE.Layer):
	"""Subsamples proposals and generates target box refinement, class_ids,
	and masks for each.
	Inputs:
	proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
			   be zero padded if there are not enough proposals.
	gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
	gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
			  coordinates.
	gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type
	Returns: Target ROIs and corresponding class IDs, bounding box shifts,
	and masks.
	rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
		  coordinates
	target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
	target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]
	target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width]
				 Masks cropped to bbox boundaries and resized to neural
				 network output size.
	Note: Returned arrays might be zero padded if not enough target ROIs.
	"""

	def __init__(self, config, **kwargs):
		super(DetectionTargetLayer, self).__init__(**kwargs)
		self.config = config

	def get_config(self):
		config = super(DetectionTargetLayer, self).get_config()
		config["config"] = self.config.to_dict()
		return config

	def call(self, inputs):
		proposals = inputs[0]
		gt_class_ids = inputs[1]
		gt_boxes = inputs[2]
		gt_masks = inputs[3]

		# Slice the batch and run a graph for each slice
		# TODO: Rename target_bbox to target_deltas for clarity
		names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
		outputs = utils.batch_slice(
			[proposals, gt_class_ids, gt_boxes, gt_masks],
			lambda w, x, y, z: detection_targets_graph(
				w, x, y, z, self.config),
			self.config.IMAGES_PER_GPU, names=names)
		return outputs

	def compute_output_shape(self, input_shape):
		return [
			(None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
			(None, self.config.TRAIN_ROIS_PER_IMAGE),  # class_ids
			(None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
			(None, self.config.TRAIN_ROIS_PER_IMAGE, self.config.MASK_SHAPE[0],
			 self.config.MASK_SHAPE[1])  # masks
		]

	def compute_mask(self, inputs, mask=None):
		return [None, None, None, None]


############################################################
#  Detection Layer
############################################################

def refine_detections_graph(rois, probs, deltas, window, config):
	"""Refine classified proposals and filter overlaps and return final
	detections.
	Inputs:
		rois: [N, (y1, x1, y2, x2)] in normalized coordinates
		probs: [N, num_classes]. Class probabilities.
		deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
				bounding box deltas.
		window: (y1, x1, y2, x2) in normalized coordinates. The part of the image
			that contains the image excluding the padding.
	Returns detections shaped: [num_detections, (y1, x1, y2, x2, class_id, score)] where
		coordinates are normalized.
	"""
	# Class IDs per ROI
	class_ids = tf.argmax(input=probs, axis=1, output_type=tf.int32)
	# Class probability of the top class of each ROI
	indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
	class_scores = tf.gather_nd(probs, indices)
	# Class-specific bounding box deltas
	deltas_specific = tf.gather_nd(deltas, indices)
	# Apply bounding box deltas
	# Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
	refined_rois = apply_box_deltas_graph(
		rois, deltas_specific * config.BBOX_STD_DEV)
	# Clip boxes to image window
	refined_rois = clip_boxes_graph(refined_rois, window)

	# TODO: Filter out boxes with zero area

	# Filter out background boxes
	keep = tf.compat.v1.where(class_ids > 0)[:, 0]
	# Filter out low confidence boxes
	if config.DETECTION_MIN_CONFIDENCE:
		conf_keep = tf.compat.v1.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
		keep = tf.sets.intersection(tf.expand_dims(keep, 0),
										tf.expand_dims(conf_keep, 0))
		keep = tf.sparse.to_dense(keep)[0]

	# Apply per-class NMS
	# 1. Prepare variables
	pre_nms_class_ids = tf.gather(class_ids, keep)
	pre_nms_scores = tf.gather(class_scores, keep)
	pre_nms_rois = tf.gather(refined_rois,   keep)
	unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

	def nms_keep_map(class_id):
		"""Apply Non-Maximum Suppression on ROIs of the given class."""
		# Indices of ROIs of the given class
		ixs = tf.compat.v1.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
		# Apply NMS
		class_keep = tf.image.non_max_suppression(
				tf.gather(pre_nms_rois, ixs),
				tf.gather(pre_nms_scores, ixs),
				max_output_size=config.DETECTION_MAX_INSTANCES,
				iou_threshold=config.DETECTION_NMS_THRESHOLD)
		# Map indices
		class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
		# Pad with -1 so returned tensors have the same shape
		gap = config.DETECTION_MAX_INSTANCES - tf.shape(input=class_keep)[0]
		class_keep = tf.pad(tensor=class_keep, paddings=[(0, gap)],
							mode='CONSTANT', constant_values=-1)
		# Set shape so map_fn() can infer result shape
		class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
		return class_keep

	# 2. Map over class IDs
	nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
						 dtype=tf.int64)
	# 3. Merge results into one list, and remove -1 padding
	nms_keep = tf.reshape(nms_keep, [-1])
	nms_keep = tf.gather(nms_keep, tf.compat.v1.where(nms_keep > -1)[:, 0])
	# 4. Compute intersection between keep and nms_keep
	keep = tf.sets.intersection(tf.expand_dims(keep, 0),
									tf.expand_dims(nms_keep, 0))
	keep = tf.sparse.to_dense(keep)[0]
	# Keep top detections
	roi_count = config.DETECTION_MAX_INSTANCES
	class_scores_keep = tf.gather(class_scores, keep)
	num_keep = tf.minimum(tf.shape(input=class_scores_keep)[0], roi_count)
	top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
	keep = tf.gather(keep, top_ids)

	# Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
	# Coordinates are normalized.
	detections = tf.concat([
		tf.gather(refined_rois, keep),
		tf.dtypes.cast(tf.gather(class_ids, keep), tf.float32)[..., tf.newaxis],
		tf.gather(class_scores, keep)[..., tf.newaxis]
		], axis=1)

	# Pad with zeros if detections < DETECTION_MAX_INSTANCES
	gap = config.DETECTION_MAX_INSTANCES - tf.shape(input=detections)[0]
	detections = tf.pad(tensor=detections, paddings=[(0, gap), (0, 0)], mode="CONSTANT")
	return detections


class DetectionLayer(KE.Layer):
	"""Takes classified proposal boxes and their bounding box deltas and
	returns the final detection boxes.
	Returns:
	[batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where
	coordinates are normalized.
	"""

	def __init__(self, config=None, **kwargs):
		super(DetectionLayer, self).__init__(**kwargs)
		self.config = config

	def get_config(self):
		config = super(DetectionLayer, self).get_config()
		config["config"] = self.config.to_dict()
		return config

	def call(self, inputs):
		rois = inputs[0]
		mrcnn_class = inputs[1]
		mrcnn_bbox = inputs[2]
		image_meta = inputs[3]

		# Get windows of images in normalized coordinates. Windows are the area
		# in the image that excludes the padding.
		# Use the shape of the first image in the batch to normalize the window
		# because we know that all images get resized to the same size.
		m = parse_image_meta_graph(image_meta)
		image_shape = m['image_shape'][0]
		window = norm_boxes_graph(m['window'], image_shape[:2])

		# Run detection refinement graph on each item in the batch
		detections_batch = utils.batch_slice(
			[rois, mrcnn_class, mrcnn_bbox, window],
			lambda x, y, w, z: refine_detections_graph(x, y, w, z, self.config),
			self.config.IMAGES_PER_GPU)

		# Reshape output
		# [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] in
		# normalized coordinates
		return tf.reshape(
			detections_batch,
			[self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6])

	def compute_output_shape(self, input_shape):
		return (None, self.config.DETECTION_MAX_INSTANCES, 6)


############################################################
#  Region Proposal Network (RPN)
############################################################

def rpn_graph(feature_map, anchors_per_location, anchor_stride):
	"""Builds the computation graph of Region Proposal Network.
	feature_map: backbone features [batch, height, width, depth]
	anchors_per_location: number of anchors per pixel in the feature map
	anchor_stride: Controls the density of anchors. Typically 1 (anchors for
				   every pixel in the feature map), or 2 (every other pixel).
	Returns:
		rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
		rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
		rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
				  applied to anchors.
	"""
	# TODO: check if stride of 2 causes alignment issues if the feature map
	# is not even.
	# Shared convolutional base of the RPN
	shared = KL.Conv2D(512, (3, 3), padding='same', activation='relu',
					   strides=anchor_stride,
					   name='rpn_conv_shared')(feature_map)

	# Anchor Score. [batch, height, width, anchors per location * 2].
	x = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
				  activation='linear', name='rpn_class_raw')(shared)

	# Reshape to [batch, anchors, 2]
	rpn_class_logits = KL.Lambda(
		lambda t: tf.reshape(t, [tf.shape(input=t)[0], -1, 2]))(x)

	# Softmax on last dimension of BG/FG.
	rpn_probs = KL.Activation(
		"softmax", name="rpn_class_xxx")(rpn_class_logits)

	# Bounding box refinement. [batch, H, W, anchors per location * depth]
	# where depth is [x, y, log(w), log(h)]
	x = KL.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
				  activation='linear', name='rpn_bbox_pred')(shared)

	# Reshape to [batch, anchors, 4]
	rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(input=t)[0], -1, 4]))(x)

	return [rpn_class_logits, rpn_probs, rpn_bbox]


def build_rpn_model(anchor_stride, anchors_per_location, depth):
	"""Builds a Keras model of the Region Proposal Network.
	It wraps the RPN graph so it can be used multiple times with shared
	weights.
	anchors_per_location: number of anchors per pixel in the feature map
	anchor_stride: Controls the density of anchors. Typically 1 (anchors for
				   every pixel in the feature map), or 2 (every other pixel).
	depth: Depth of the backbone feature map.
	Returns a Keras Model object. The model outputs, when called, are:
	rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
	rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
	rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
				applied to anchors.
	"""
	input_feature_map = KL.Input(shape=[None, None, depth],
								 name="input_rpn_feature_map")
	outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
	return KM.Model([input_feature_map], outputs, name="rpn_model")


############################################################
#  Feature Pyramid Network Heads
############################################################

def fpn_classifier_graph(rois, feature_maps, image_meta,
						 pool_size, num_classes, train_bn=True,
						 fc_layers_size=1024):
	"""Builds the computation graph of the feature pyramid network classifier
	and regressor heads.
	rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
		  coordinates.
	feature_maps: List of feature maps from different layers of the pyramid,
				  [P2, P3, P4, P5]. Each has a different resolution.
	image_meta: [batch, (meta data)] Image details. See compose_image_meta()
	pool_size: The width of the square feature map generated from ROI Pooling.
	num_classes: number of classes, which determines the depth of the results
	train_bn: Boolean. Train or freeze Batch Norm layers
	fc_layers_size: Size of the 2 FC layers
	Returns:
		logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
		probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
		bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] Deltas to apply to
					 proposal boxes
	"""
	# ROI Pooling
	# Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
	x = PyramidROIAlign([pool_size, pool_size],
						name="roi_align_classifier")([rois, image_meta] + feature_maps)
	# Two 1024 FC layers (implemented with Conv2D for consistency)
	x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"),
						   name="mrcnn_class_conv1")(x)
	x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn1')(x, training=train_bn)
	x = KL.Activation('relu')(x)
	x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (1, 1)),
						   name="mrcnn_class_conv2")(x)
	x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn2')(x, training=train_bn)
	x = KL.Activation('relu')(x)

	shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),
					   name="pool_squeeze")(x)

	# Classifier head
	mrcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes),
											name='mrcnn_class_logits')(shared)
	mrcnn_probs = KL.TimeDistributed(KL.Activation("softmax"),
									 name="mrcnn_class")(mrcnn_class_logits)

	# BBox head
	# [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
	x = KL.TimeDistributed(KL.Dense(num_classes * 4, activation='linear'),
						   name='mrcnn_bbox_fc')(shared)
	# Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
	s = K.int_shape(x)
	if s[1] is None:
		mrcnn_bbox = KL.Reshape((-1, num_classes, 4), name="mrcnn_bbox")(x)
	else:
		mrcnn_bbox = KL.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)

	return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def build_fpn_mask_graph(rois, feature_maps, image_meta,
						 pool_size, num_classes, train_bn=True):
	"""Builds the computation graph of the mask head of Feature Pyramid Network.
	rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
		  coordinates.
	feature_maps: List of feature maps from different layers of the pyramid,
				  [P2, P3, P4, P5]. Each has a different resolution.
	image_meta: [batch, (meta data)] Image details. See compose_image_meta()
	pool_size: The width of the square feature map generated from ROI Pooling.
	num_classes: number of classes, which determines the depth of the results
	train_bn: Boolean. Train or freeze Batch Norm layers
	Returns: Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
	"""
	# ROI Pooling
	# Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
	x = PyramidROIAlign([pool_size, pool_size],
						name="roi_align_mask")([rois, image_meta] + feature_maps)

	# Conv layers
	x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
						   name="mrcnn_mask_conv1")(x)
	x = KL.TimeDistributed(BatchNorm(),
						   name='mrcnn_mask_bn1')(x, training=train_bn)
	x = KL.Activation('relu')(x)

	x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
						   name="mrcnn_mask_conv2")(x)
	x = KL.TimeDistributed(BatchNorm(),
						   name='mrcnn_mask_bn2')(x, training=train_bn)
	x = KL.Activation('relu')(x)

	x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
						   name="mrcnn_mask_conv3")(x)
	x = KL.TimeDistributed(BatchNorm(),
						   name='mrcnn_mask_bn3')(x, training=train_bn)
	x = KL.Activation('relu')(x)

	x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
						   name="mrcnn_mask_conv4")(x)
	x = KL.TimeDistributed(BatchNorm(),
						   name='mrcnn_mask_bn4')(x, training=train_bn)
	x = KL.Activation('relu')(x)

	x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
						   name="mrcnn_mask_deconv")(x)
	x = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"),
						   name="mrcnn_mask")(x)
	return x


############################################################
#  Loss Functions
############################################################

def smooth_l1_loss(y_true, y_pred):
	"""Implements Smooth-L1 loss.
	y_true and y_pred are typically: [N, 4], but could be any shape.
	"""
	diff = K.abs(y_true - y_pred)
	less_than_one = K.cast(K.less(diff, 1.0), "float32")
	loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
	return loss


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
	"""RPN anchor classifier loss.
	rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
			   -1=negative, 0=neutral anchor.
	rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for BG/FG.
	"""
	# Squeeze last dim to simplify
	rpn_match = tf.squeeze(rpn_match, -1)
	# Get anchor classes. Convert the -1/+1 match to 0/1 values.
	anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)
	# Positive and Negative anchors contribute to the loss,
	# but neutral anchors (match value = 0) don't.
	indices = tf.compat.v1.where(K.not_equal(rpn_match, 0))
	# Pick rows that contribute to the loss and filter out the rest.
	rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
	anchor_class = tf.gather_nd(anchor_class, indices)
	# Cross entropy loss
	loss = K.sparse_categorical_crossentropy(target=anchor_class,
											 output=rpn_class_logits,
											 from_logits=True)
	loss = K.switch(tf.size(input=loss) > 0, K.mean(loss), tf.constant(0.0))
	return loss


def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
	"""Return the RPN bounding box loss graph.
	config: the model config object.
	target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
		Uses 0 padding to fill in unsed bbox deltas.
	rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
			   -1=negative, 0=neutral anchor.
	rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
	"""
	# Positive anchors contribute to the loss, but negative and
	# neutral anchors (match value of 0 or -1) don't.
	rpn_match = K.squeeze(rpn_match, -1)
	indices = tf.compat.v1.where(K.equal(rpn_match, 1))

	# Pick bbox deltas that contribute to the loss
	rpn_bbox = tf.gather_nd(rpn_bbox, indices)

	# Trim target bounding box deltas to the same length as rpn_bbox.
	batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
	target_bbox = batch_pack_graph(target_bbox, batch_counts,
								   config.IMAGES_PER_GPU)

	loss = smooth_l1_loss(target_bbox, rpn_bbox)

	loss = K.switch(tf.size(input=loss) > 0, K.mean(loss), tf.constant(0.0))
	return loss


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits,
						   active_class_ids):
	"""Loss for the classifier head of Mask RCNN.
	target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
		padding to fill in the array.
	pred_class_logits: [batch, num_rois, num_classes]
	active_class_ids: [batch, num_classes]. Has a value of 1 for
		classes that are in the dataset of the image, and 0
		for classes that are not in the dataset.
	"""
	# During model building, Keras calls this function with
	# target_class_ids of type float32. Unclear why. Cast it
	# to int to get around it.
	target_class_ids = tf.cast(target_class_ids, 'int64')

	# Find predictions of classes that are not in the dataset.
	pred_class_ids = tf.argmax(input=pred_class_logits, axis=2)
	# TODO: Update this line to work with batch > 1. Right now it assumes all
	#       images in a batch have the same active_class_ids
	pred_active = tf.gather(active_class_ids[0], pred_class_ids)

	# Loss
	loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
		labels=target_class_ids, logits=pred_class_logits)

	# Erase losses of predictions of classes that are not in the active
	# classes of the image.
	loss = loss * pred_active

	# Computer loss mean. Use only predictions that contribute
	# to the loss to get a correct mean.
	loss = tf.reduce_sum(input_tensor=loss) / tf.reduce_sum(input_tensor=pred_active)
	return loss


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
	"""Loss for Mask R-CNN bounding box refinement.
	target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
	target_class_ids: [batch, num_rois]. Integer class IDs.
	pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
	"""
	# Reshape to merge batch and roi dimensions for simplicity.
	target_class_ids = K.reshape(target_class_ids, (-1,))
	target_bbox = K.reshape(target_bbox, (-1, 4))
	pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 4))

	# Only positive ROIs contribute to the loss. And only
	# the right class_id of each ROI. Get their indices.
	positive_roi_ix = tf.compat.v1.where(target_class_ids > 0)[:, 0]
	positive_roi_class_ids = tf.cast(
		tf.gather(target_class_ids, positive_roi_ix), tf.int64)
	indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

	# Gather the deltas (predicted and true) that contribute to loss
	target_bbox = tf.gather(target_bbox, positive_roi_ix)
	pred_bbox = tf.gather_nd(pred_bbox, indices)

	# Smooth-L1 Loss
	loss = K.switch(tf.size(input=target_bbox) > 0,
					smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
					tf.constant(0.0))
	loss = K.mean(loss)
	return loss


def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
	"""Mask binary cross-entropy loss for the masks head.
	target_masks: [batch, num_rois, height, width].
		A float32 tensor of values 0 or 1. Uses zero padding to fill array.
	target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
	pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
				with values from 0 to 1.
	"""
	# Reshape for simplicity. Merge first two dimensions into one.
	target_class_ids = K.reshape(target_class_ids, (-1,))
	mask_shape = tf.shape(input=target_masks)
	target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
	pred_shape = tf.shape(input=pred_masks)
	pred_masks = K.reshape(pred_masks,
						   (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
	# Permute predicted masks to [N, num_classes, height, width]
	pred_masks = tf.transpose(a=pred_masks, perm=[0, 3, 1, 2])

	# Only positive ROIs contribute to the loss. And only
	# the class specific mask of each ROI.
	positive_ix = tf.compat.v1.where(target_class_ids > 0)[:, 0]
	positive_class_ids = tf.cast(
		tf.gather(target_class_ids, positive_ix), tf.int64)
	indices = tf.stack([positive_ix, positive_class_ids], axis=1)

	# Gather the masks (predicted and true) that contribute to loss
	y_true = tf.gather(target_masks, positive_ix)
	y_pred = tf.gather_nd(pred_masks, indices)

	# Compute binary cross entropy. If no positive ROIs, then return 0.
	# shape: [batch, roi, num_classes]
	loss = K.switch(tf.size(input=y_true) > 0,
					K.binary_crossentropy(target=y_true, output=y_pred),
					tf.constant(0.0))
	loss = K.mean(loss)
	return loss


############################################################
#  Data Generator
############################################################

def load_image_gt(dataset, config, image_id, augment=False, augmentation=None,
				  use_mini_mask=False):
	"""Load and return ground truth data for an image (image, mask, bounding boxes).
	augment: (deprecated. Use augmentation instead). If true, apply random
		image augmentation. Currently, only horizontal flipping is offered.
	augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
		For example, passing imgaug.augmenters.Fliplr(0.5) flips images
		right/left 50% of the time.
	use_mini_mask: If False, returns full-size masks that are the same height
		and width as the original image. These can be big, for example
		1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
		224x224 and are generated by extracting the bounding box of the
		object and resizing it to MINI_MASK_SHAPE.
	Returns:
	image: [height, width, 3]
	shape: the original shape of the image before resizing and cropping.
	class_ids: [instance_count] Integer class IDs
	bbox: [instance_count, (y1, x1, y2, x2)]
	mask: [height, width, instance_count]. The height and width are those
		of the image unless use_mini_mask is True, in which case they are
		defined in MINI_MASK_SHAPE.
	"""
	# Load image and mask
	mask, class_ids = dataset.load_mask(image_id)
	image = dataset.load_image(image_id, config.INPUT_DIM, mask=mask, z_to=config.INPUT_Z)
	original_shape = image.shape
	image, window, scale, padding, crop = utils.resize_image(image,
											min_dim=config.IMAGE_MIN_DIM,
											min_scale=config.IMAGE_MIN_SCALE,
											max_dim=config.IMAGE_MAX_DIM,
											mode=config.IMAGE_RESIZE_MODE)

	mask = utils.resize_mask(mask, scale, padding, crop)

	# Random horizontal flips.
	# TODO: will be removed in a future update in favor of augmentation
	if augment:
		logging.warning("'augment' is deprecated. Use 'augmentation' instead.")
		if random.randint(0, 1):
			image = np.fliplr(image)
			mask = np.fliplr(mask)

	# Augmentation
	# This requires the imgaug lib (https://github.com/aleju/imgaug)
	if augmentation:
		import imgaug
		import imgaug.augmenters as iaa
		#imgaug requires :
		# 'images' should be either a 4D numpy array of shape (N, height, width, channels)
		# or a list of 3D numpy arrays, each having shape (height, width, channels).
		# Grayscale images must have shape (height, width, 1) each.
		# All images must have numpy's dtype uint8. Values are expected to be in
		# range 0-255 integers.
		#(H,W,C) ndarray or (H,W) ndarray
		#so roll axis, then roll back?

		#from cell.py __main__
		#clever idea: determine probabilities of selecting particular augmentors BEFORE.
		#this will need to go into: model.py BEFORE MASK_AUGMENTORS line.
		#set up a list of augmentors.
		#see imgaug lambda for adding your own new augmentation, if necessary.
		#select one of these:
		affine_list = [iaa.Affine(rotate=90,name="rot90"),
					   iaa.Affine(rotate=180,name="rot180"),
					   iaa.Affine(rotate=270,name="rot270")]
		N = np.random.rand(1,len(affine_list))[0]
		N = N/np.sum(N)
		N = N.tolist()
		#pick affine
		affine_pick = affine_list[N.index(max(N))]
		##randomx shear
		#xsheardeg = np.random.randint(-45,45)
		#ysheardeg = np.random.randint(-45,45)
		#shearing is actually not a great idea. It resizes objects.
		augment_list = [iaa.Fliplr(1,name="fliplr"),
						iaa.Flipud(1,name="flipud"),
						affine_pick]#,
						#iaa.Affine(shear={'x': xsheardeg, 'y': ysheardeg},name="sheared")]
		#probability any one of these will be used is:
		P_use = 0.5
		#pick N (length of augment_list) random numbers.
		N = np.random.rand(1,len(augment_list))[0]
		#take indices where N > P_use
		inds = [i for i,x in enumerate(N) if x<=P_use]
		augment_used = [x for i,x in enumerate(augment_list) if i in inds]

		aug = iaa.Sequential(augment_used)


		det = aug.to_deterministic()

		MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
						   "Fliplr", "Flipud", "CropAndPad",
						   "Affine", "PiecewiseAffine"]

		def hook(images, augmenter, parents, default):
			"""Determines which augmenters to apply to masks."""
			return augmenter.__class__.__name__ in MASK_AUGMENTERS

		# Store shapes before augmentation to compare
		image_shape = image.shape
		mask_shape = mask.shape
		#print(image.shape)
		#print(mask.shape)
		import warnings
		warnings.filterwarnings('ignore')#,".*SuspiciousSingleImageShapeWarning.*")
		#roll axis
		image = np.rollaxis(image, 2, 0) #this may flip the image...
		#image_rot = np.rollaxis(image, 1, 3)
		image = det.augment_images(image)
		mask = det.augment_image(mask.astype(np.uint8),
								 hooks=imgaug.HooksImages(activator=hook))
		#roll axis back
		image = np.rollaxis(image,0, 3)

		#flip image augmentation if 3D
		if config.INPUT_DIM=="3D":
			if np.random.rand()>0.5:
				image = np.flip(image,axis=2)

		assert image.shape == image_shape, "Augmentation shouldn't change image size"
		assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
		# Change mask back to bool
		mask = mask.astype(np.bool)
		#now, since it was augmented, sometimes the masks will not be part of the image.
		#Go through the images of mask, delete those that have all zeros

	# Note that some boxes might be all zeros if the corresponding mask got cropped out.
	# and here is to filter them out
	_idx = np.sum(mask, axis=(0, 1)) > 0
	mask = mask[:, :, _idx]
	class_ids = class_ids[_idx]
	# Bounding boxes. Note that some boxes might be all zeros
	# if the corresponding mask got cropped out.
	# bbox: [num_instances, (y1, x1, y2, x2)]
	bbox = utils.extract_bboxes(mask)

	# Active classes
	# Different datasets have different classes, so track the
	# classes supported in the dataset of this image.
	active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
	source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
	active_class_ids[source_class_ids] = 1

	# Resize masks to smaller size to reduce memory usage
	if use_mini_mask:
		mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

	# Image meta data
	image_meta = compose_image_meta(image_id, original_shape, image.shape,
									window, scale, active_class_ids)

	return image, image_meta, class_ids, bbox, mask

def load_image_gt2(dataset, config, image_id, augment=False, augmentation=None,
				  use_mini_mask=False):
	"""Load and return ground truth data for an image (image, mask, bounding boxes).
	augment: (deprecated. Use augmentation instead). If true, apply random
		image augmentation. Currently, only horizontal flipping is offered.
	augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
		For example, passing imgaug.augmenters.Fliplr(0.5) flips images
		right/left 50% of the time.
	use_mini_mask: If False, returns full-size masks that are the same height
		and width as the original image. These can be big, for example
		1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
		224x224 and are generated by extracting the bounding box of the
		object and resizing it to MINI_MASK_SHAPE.
	Returns:
	image: [height, width, 3]
	shape: the original shape of the image before resizing and cropping.
	class_ids: [instance_count] Integer class IDs
	bbox: [instance_count, (y1, x1, y2, x2)]
	mask: [height, width, instance_count]. The height and width are those
		of the image unless use_mini_mask is True, in which case they are
		defined in MINI_MASK_SHAPE.
	###########################################################################
	#This method reduces the computation time by 38% compared to load_image_gt#
	###########################################################################
	"""
	############################################################################
	####################### LOAD IMAGE AND MASK ################################
	############################################################################
	# Load image and mask
	mask, class_ids = dataset.load_mask(image_id)
	image = dataset.load_image(image_id, config.INPUT_DIM, mask=mask, z_to=config.INPUT_Z)
	cents, zlocs, imshape = dataset.load_z_positions(image_id)
	#switch the rows, colums of cents.
	cents = cents[:,::-1]

	original_shape = image.shape

	###########################################################################
	############################### CROP ######################################
	###########################################################################
	""" Now, we will crop the image/mask pair to appropriate shape """
	#Make necessary copies
	image_full = copy.deepcopy(image)
	mask_full = copy.deepcopy(mask)
	zlocs_full = copy.deepcopy(zlocs)
	cents_full = copy.deepcopy(cents)

	good_load = False
	while not good_load:
		#Do the resizing
		image, window, scale, padding, crop = utils.resize_image(
							image_full, cents, min_dim=config.IMAGE_MIN_DIM, \
							max_dim=config.IMAGE_MAX_DIM, \
							min_scale=config.IMAGE_MIN_SCALE, \
							mode=config.IMAGE_RESIZE_MODE)

		mask = utils.resize_mask(mask_full, scale, padding, crop=crop)
		#Go through the images of mask, delete those that have all zeros
		#filter 1
		_idx1 = np.sum(mask, axis=(0, 1)) > 10
		"""
		What do we do if mask is totally empty, that is a segment was picked that has nothing in it?
		One easy option is just to return zeros for mask, xgrad, and ygrad.
		"""
		if any(np.ravel(_idx1)):
			good_load=True
			mask = mask[:, :, _idx1]
			zlocs = zlocs_full[_idx1]
			cents = cents_full[_idx1,:]
			class_ids = class_ids[_idx1]

	##########################################################################
	######################### APPLY AUGMENTATIONS ############################
	##########################################################################
	# Random horizontal flips.
	# TODO: will be removed in a future update in favor of augmentation
	if augment:
		logging.warning("'augment' is deprecated. Use 'augmentation' instead.")
		if random.randint(0, 1):
			image, mask, cents, _ = image_xyflip(image, mask, cents, 0.5)

	# Augmentation
	# This requires the imgaug lib (https://github.com/aleju/imgaug)
	if augmentation:
		import imgaug
		import imgaug.augmenters as iaa
		#imgaug requires :
		# 'images' should be either a 4D numpy array of shape (N, height, width, channels)
		# or a list of 3D numpy arrays, each having shape (height, width, channels).
		# Grayscale images must have shape (height, width, 1) each.
		# All images must have numpy's dtype uint8. Values are expected to be in
		# range 0-255 integers.
		#(H,W,C) ndarray or (H,W) ndarray
		#so roll axis, then roll back?

		#from cell.py __main__
		#clever idea: determine probabilities of selecting particular augmentors BEFORE.
		#this will need to go into: model.py BEFORE MASK_AUGMENTORS line.
		#set up a list of augmentors.
		#see imgaug lambda for adding your own new augmentation, if necessary.
		#select one of these:
		#apply x-y flip
		image, mask, cents, _ = image_xyflip(image, mask, cents, 0.5)
		#apply rotations
		image, mask, cents, _ = image_mask_rotate(image, mask, cents, 0.5)

		#apply z-flip, dense and translate augmentations
		###############################################################
		###############################################################
		"""
		NEW! In order to allow the images to be made more object dense,
		I have implemented algorithms to repeat objects in an image.
		TIMING: These operations take less than 1 second to complete for
		an image that is 1024x1024x27x1.
		The part that takes much longer is now operations on the masks,
		which go from H x W x N to at most H x W x 2N.
		"""
		#we have to do these operations on the whole image
		#in order for the z-class assignment to be reproducible!
		#flip image in z-dimension
		if config.INPUT_DIM=="3D":
			if np.random.rand()>=0.5:
				image = image[:,:,::-1,:]
				#flip zlocs.
				zlocs = (image.shape[2]-1) - zlocs

		if config.INPUT_DIM=="3D":
			#try to stack
			stacked_var=1
			if int(np.floor(config.INPUT_Z / image.shape[2]))>1:
				stacked_var+=1
				image, mask, zlocs, cents = stack_augmentation(image, mask, zlocs, cents, config.INPUT_Z)

		if config.INPUT_DIM=="3D":
			#could repeat this operation N times to make it dense.
			#we don't want to do this operation if we've stacked the image,
			#might make it too dense.
			if stacked_var==1:
				image, mask, zlocs, cents = dense_translate_augmentation(image, mask, zlocs, cents, config.INPUT_Z, config.OUT_CHANNELS)
		###############################################################
		###############################################################


		# Change mask back to bool
		mask = mask>0.5
		#now, since it was augmented, sometimes the masks will not be part of the image.
		#Go through the images of mask, delete those that have all zeros

	#because of the augmentations above, there might be N times more masks than before.
	# in this case, figure out how many class_ids to add.
	if mask.shape[2]>len(class_ids):
		#by how much.
		class_ids = np.repeat(class_ids, int(mask.shape[2]/len(class_ids)))
	# Note that some boxes might be all zeros if the corresponding mask got cropped out.
	# and here is to filter them out
	_idx = np.sum(mask, axis=(0, 1)) > 0
	mask = mask[:, :, _idx]
	class_ids = class_ids[_idx]
	# Bounding boxes. Note that some boxes might be all zeros
	# if the corresponding mask got cropped out.
	# bbox: [num_instances, (y1, x1, y2, x2)]
	bbox = utils.extract_bboxes(mask)

	#########################################################################
	######################### PAD THE IMAGE #################################
	#########################################################################
	#pad the images
	if config.INPUT_DIM=="3D":
		image = dataset.pad_z_image(image, z_to = config.INPUT_Z, center = True, random_pad = False)


	# Active classes
	# Different datasets have different classes, so track the
	# classes supported in the dataset of this image.
	active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
	source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
	active_class_ids[source_class_ids] = 1

	# Resize masks to smaller size to reduce memory usage
	if use_mini_mask:
		mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

	# Image meta data
	image_meta = compose_image_meta(image_id, original_shape, image.shape,
									window, scale, active_class_ids)

	return image, image_meta, class_ids, bbox, mask

def load_image_gt_metrics(dataset, config, image_id):
	"""Load and return ground truth data for an image (image, mask, bounding boxes).

	Returns:
	gt_masks: [height, width, instance_count]. The height and width are those
		of the image.
	"""
	gt_masks, _ = dataset.load_mask(image_id)
	pred_masks = dataset.load_detection(image_id)
	return gt_masks, pred_masks


def build_detection_targets(rpn_rois, gt_class_ids, gt_boxes, gt_masks, config):
	"""Generate targets for training Stage 2 classifier and mask heads.
	This is not used in normal training. It's useful for debugging or to train
	the Mask RCNN heads without using the RPN head.
	Inputs:
	rpn_rois: [N, (y1, x1, y2, x2)] proposal boxes.
	gt_class_ids: [instance count] Integer class IDs
	gt_boxes: [instance count, (y1, x1, y2, x2)]
	gt_masks: [height, width, instance count] Ground truth masks. Can be full
			  size or mini-masks.
	Returns:
	rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]
	class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
	bboxes: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (y, x, log(h), log(w))]. Class-specific
			bbox refinements.
	masks: [TRAIN_ROIS_PER_IMAGE, height, width, NUM_CLASSES). Class specific masks cropped
		   to bbox boundaries and resized to neural network output size.
	"""
	assert rpn_rois.shape[0] > 0
	assert gt_class_ids.dtype == np.int32, "Expected int but got {}".format(
		gt_class_ids.dtype)
	assert gt_boxes.dtype == np.int32, "Expected int but got {}".format(
		gt_boxes.dtype)
	assert gt_masks.dtype == np.bool_, "Expected bool but got {}".format(
		gt_masks.dtype)

	# It's common to add GT Boxes to ROIs but we don't do that here because
	# according to XinLei Chen's paper, it doesn't help.

	# Trim empty padding in gt_boxes and gt_masks parts
	instance_ids = np.where(gt_class_ids > 0)[0]
	assert instance_ids.shape[0] > 0, "Image must contain instances."
	gt_class_ids = gt_class_ids[instance_ids]
	gt_boxes = gt_boxes[instance_ids]
	gt_masks = gt_masks[:, :, instance_ids]

	# Compute areas of ROIs and ground truth boxes.
	rpn_roi_area = (rpn_rois[:, 2] - rpn_rois[:, 0]) * \
		(rpn_rois[:, 3] - rpn_rois[:, 1])
	gt_box_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * \
		(gt_boxes[:, 3] - gt_boxes[:, 1])

	# Compute overlaps [rpn_rois, gt_boxes]
	overlaps = np.zeros((rpn_rois.shape[0], gt_boxes.shape[0]))
	for i in range(overlaps.shape[1]):
		gt = gt_boxes[i]
		overlaps[:, i] = utils.compute_iou(
			gt, rpn_rois, gt_box_area[i], rpn_roi_area)

	# Assign ROIs to GT boxes
	rpn_roi_iou_argmax = np.argmax(overlaps, axis=1)
	rpn_roi_iou_max = overlaps[np.arange(
		overlaps.shape[0]), rpn_roi_iou_argmax]
	# GT box assigned to each ROI
	rpn_roi_gt_boxes = gt_boxes[rpn_roi_iou_argmax]
	rpn_roi_gt_class_ids = gt_class_ids[rpn_roi_iou_argmax]

	# Positive ROIs are those with >= 0.5 IoU with a GT box.
	fg_ids = np.where(rpn_roi_iou_max > 0.5)[0]

	# Negative ROIs are those with max IoU 0.1-0.5 (hard example mining)
	# TODO: To hard example mine or not to hard example mine, that's the question
	# bg_ids = np.where((rpn_roi_iou_max >= 0.1) & (rpn_roi_iou_max < 0.5))[0]
	bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]

	# Subsample ROIs. Aim for 33% foreground.
	# FG
	fg_roi_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
	if fg_ids.shape[0] > fg_roi_count:
		keep_fg_ids = np.random.choice(fg_ids, fg_roi_count, replace=False)
	else:
		keep_fg_ids = fg_ids
	# BG
	remaining = config.TRAIN_ROIS_PER_IMAGE - keep_fg_ids.shape[0]
	if bg_ids.shape[0] > remaining:
		keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
	else:
		keep_bg_ids = bg_ids
	# Combine indices of ROIs to keep
	keep = np.concatenate([keep_fg_ids, keep_bg_ids])
	# Need more?
	remaining = config.TRAIN_ROIS_PER_IMAGE - keep.shape[0]
	if remaining > 0:
		# Looks like we don't have enough samples to maintain the desired
		# balance. Reduce requirements and fill in the rest. This is
		# likely different from the Mask RCNN paper.

		# There is a small chance we have neither fg nor bg samples.
		if keep.shape[0] == 0:
			# Pick bg regions with easier IoU threshold
			bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]
			assert bg_ids.shape[0] >= remaining
			keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
			assert keep_bg_ids.shape[0] == remaining
			keep = np.concatenate([keep, keep_bg_ids])
		else:
			# Fill the rest with repeated bg rois.
			keep_extra_ids = np.random.choice(
				keep_bg_ids, remaining, replace=True)
			keep = np.concatenate([keep, keep_extra_ids])
	assert keep.shape[0] == config.TRAIN_ROIS_PER_IMAGE, \
		"keep doesn't match ROI batch size {}, {}".format(
			keep.shape[0], config.TRAIN_ROIS_PER_IMAGE)

	# Reset the gt boxes assigned to BG ROIs.
	rpn_roi_gt_boxes[keep_bg_ids, :] = 0
	rpn_roi_gt_class_ids[keep_bg_ids] = 0

	# For each kept ROI, assign a class_id, and for FG ROIs also add bbox refinement.
	rois = rpn_rois[keep]
	roi_gt_boxes = rpn_roi_gt_boxes[keep]
	roi_gt_class_ids = rpn_roi_gt_class_ids[keep]
	roi_gt_assignment = rpn_roi_iou_argmax[keep]

	# Class-aware bbox deltas. [y, x, log(h), log(w)]
	bboxes = np.zeros((config.TRAIN_ROIS_PER_IMAGE,
					   config.NUM_CLASSES, 4), dtype=np.float32)
	pos_ids = np.where(roi_gt_class_ids > 0)[0]
	bboxes[pos_ids, roi_gt_class_ids[pos_ids]] = utils.box_refinement(
		rois[pos_ids], roi_gt_boxes[pos_ids, :4])
	# Normalize bbox refinements
	bboxes /= config.BBOX_STD_DEV

	# Generate class-specific target masks
	masks = np.zeros((config.TRAIN_ROIS_PER_IMAGE, config.MASK_SHAPE[0], config.MASK_SHAPE[1], config.NUM_CLASSES),
					 dtype=np.float32)
	for i in pos_ids:
		class_id = roi_gt_class_ids[i]
		assert class_id > 0, "class id must be greater than 0"
		gt_id = roi_gt_assignment[i]
		class_mask = gt_masks[:, :, gt_id]

		if config.USE_MINI_MASK:
			# Create a mask placeholder, the size of the image
			placeholder = np.zeros(config.IMAGE_SHAPE[:2], dtype=bool)
			# GT box
			gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[gt_id]
			gt_w = gt_x2 - gt_x1
			gt_h = gt_y2 - gt_y1
			# Resize mini mask to size of GT box
			placeholder[gt_y1:gt_y2, gt_x1:gt_x2] = \
				np.round(utils.resize(class_mask, (gt_h, gt_w))).astype(bool)
			# Place the mini batch in the placeholder
			class_mask = placeholder

		# Pick part of the mask and resize it
		y1, x1, y2, x2 = rois[i].astype(np.int32)
		m = class_mask[y1:y2, x1:x2]
		mask = utils.resize(m, config.MASK_SHAPE)
		masks[i, :, :, class_id] = mask

	return rois, roi_gt_class_ids, bboxes, masks


def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
	"""Given the anchors and GT boxes, compute overlaps and identify positive
	anchors and deltas to refine them to match their corresponding GT boxes.
	anchors: [num_anchors, (y1, x1, y2, x2)]
	gt_class_ids: [num_gt_boxes] Integer class IDs.
	gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]
	Returns:
	rpn_match: [N] (int32) matches between anchors and GT boxes.
			   1 = positive anchor, -1 = negative anchor, 0 = neutral
	rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
	"""
	# RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
	rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
	# RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
	rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

	# Handle COCO crowds
	# A crowd box in COCO is a bounding box around several instances. Exclude
	# them from training. A crowd box is given a negative class ID.
	crowd_ix = np.where(gt_class_ids < 0)[0]
	if crowd_ix.shape[0] > 0:
		# Filter out crowds from ground truth class IDs and boxes
		non_crowd_ix = np.where(gt_class_ids > 0)[0]
		crowd_boxes = gt_boxes[crowd_ix]
		gt_class_ids = gt_class_ids[non_crowd_ix]
		gt_boxes = gt_boxes[non_crowd_ix]
		# Compute overlaps with crowd boxes [anchors, crowds]
		crowd_overlaps = utils.compute_overlaps(anchors, crowd_boxes)
		crowd_iou_max = np.amax(crowd_overlaps, axis=1)
		no_crowd_bool = (crowd_iou_max < 0.001)
	else:
		# All anchors don't intersect a crowd
		no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

	# Compute overlaps [num_anchors, num_gt_boxes]
	overlaps = utils.compute_overlaps(anchors, gt_boxes)

	# Match anchors to GT Boxes
	# If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
	# If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
	# Neutral anchors are those that don't match the conditions above,
	# and they don't influence the loss function.
	# However, don't keep any GT box unmatched (rare, but happens). Instead,
	# match it to the closest anchor (even if its max IoU is < 0.3).
	#
	# 1. Set negative anchors first. They get overwritten below if a GT box is
	# matched to them. Skip boxes in crowd areas.
	anchor_iou_argmax = np.argmax(overlaps, axis=1)
	anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
	rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
	# 2. Set an anchor for each GT box (regardless of IoU value).
	# If multiple anchors have the same IoU match all of them
	gt_iou_argmax = np.argwhere(overlaps == np.max(overlaps, axis=0))[:,0]
	rpn_match[gt_iou_argmax] = 1
	# 3. Set anchors with high overlap as positive.
	rpn_match[anchor_iou_max >= 0.7] = 1

	# Subsample to balance positive and negative anchors
	# Don't let positives be more than half the anchors
	ids = np.where(rpn_match == 1)[0]
	extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
	if extra > 0:
		# Reset the extra ones to neutral
		ids = np.random.choice(ids, extra, replace=False)
		rpn_match[ids] = 0
	# Same for negative proposals
	ids = np.where(rpn_match == -1)[0]
	extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
						np.sum(rpn_match == 1))
	if extra > 0:
		# Rest the extra ones to neutral
		ids = np.random.choice(ids, extra, replace=False)
		rpn_match[ids] = 0

	# For positive anchors, compute shift and scale needed to transform them
	# to match the corresponding GT boxes.
	ids = np.where(rpn_match == 1)[0]
	ix = 0  # index into rpn_bbox
	# TODO: use box_refinement() rather than duplicating the code here
	for i, a in zip(ids, anchors[ids]):
		# Closest gt box (it might have IoU < 0.7)
		gt = gt_boxes[anchor_iou_argmax[i]]

		# Convert coordinates to center plus width/height.
		# GT Box
		gt_h = gt[2] - gt[0]
		gt_w = gt[3] - gt[1]
		gt_center_y = gt[0] + 0.5 * gt_h
		gt_center_x = gt[1] + 0.5 * gt_w
		# Anchor
		a_h = a[2] - a[0]
		a_w = a[3] - a[1]
		a_center_y = a[0] + 0.5 * a_h
		a_center_x = a[1] + 0.5 * a_w

		# Compute the bbox refinement that the RPN should predict.
		rpn_bbox[ix] = [
			(gt_center_y - a_center_y) / a_h,
			(gt_center_x - a_center_x) / a_w,
			np.log(gt_h / a_h),
			np.log(gt_w / a_w),
		]
		# Normalize
		rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
		ix += 1

	return rpn_match, rpn_bbox


def generate_random_rois(image_shape, count, gt_class_ids, gt_boxes):
	"""Generates ROI proposals similar to what a region proposal network
	would generate.
	image_shape: [Height, Width, Depth]
	count: Number of ROIs to generate
	gt_class_ids: [N] Integer ground truth class IDs
	gt_boxes: [N, (y1, x1, y2, x2)] Ground truth boxes in pixels.
	Returns: [count, (y1, x1, y2, x2)] ROI boxes in pixels.
	"""
	# placeholder
	rois = np.zeros((count, 4), dtype=np.int32)

	# Generate random ROIs around GT boxes (90% of count)
	rois_per_box = int(0.9 * count / gt_boxes.shape[0])
	for i in range(gt_boxes.shape[0]):
		gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[i]
		h = gt_y2 - gt_y1
		w = gt_x2 - gt_x1
		# random boundaries
		r_y1 = max(gt_y1 - h, 0)
		r_y2 = min(gt_y2 + h, image_shape[0])
		r_x1 = max(gt_x1 - w, 0)
		r_x2 = min(gt_x2 + w, image_shape[1])

		# To avoid generating boxes with zero area, we generate double what
		# we need and filter out the extra. If we get fewer valid boxes
		# than we need, we loop and try again.
		while True:
			y1y2 = np.random.randint(r_y1, r_y2, (rois_per_box * 2, 2))
			x1x2 = np.random.randint(r_x1, r_x2, (rois_per_box * 2, 2))
			# Filter out zero area boxes
			threshold = 1
			y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
						threshold][:rois_per_box]
			x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
						threshold][:rois_per_box]
			if y1y2.shape[0] == rois_per_box and x1x2.shape[0] == rois_per_box:
				break

		# Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
		# into x1, y1, x2, y2 order
		x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
		y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
		box_rois = np.hstack([y1, x1, y2, x2])
		rois[rois_per_box * i:rois_per_box * (i + 1)] = box_rois

	# Generate random ROIs anywhere in the image (10% of count)
	remaining_count = count - (rois_per_box * gt_boxes.shape[0])
	# To avoid generating boxes with zero area, we generate double what
	# we need and filter out the extra. If we get fewer valid boxes
	# than we need, we loop and try again.
	while True:
		y1y2 = np.random.randint(0, image_shape[0], (remaining_count * 2, 2))
		x1x2 = np.random.randint(0, image_shape[1], (remaining_count * 2, 2))
		# Filter out zero area boxes
		threshold = 1
		y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
					threshold][:remaining_count]
		x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
					threshold][:remaining_count]
		if y1y2.shape[0] == remaining_count and x1x2.shape[0] == remaining_count:
			break

	# Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
	# into x1, y1, x2, y2 order
	x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
	y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
	global_rois = np.hstack([y1, x1, y2, x2])
	rois[-remaining_count:] = global_rois
	return rois


def data_generator(dataset, config, shuffle=True, augment=False, augmentation=None,
				   random_rois=0, batch_size=1, detection_targets=False,
				   no_augmentation_sources=None):
	"""A generator that returns images and corresponding target class ids,
	bounding box deltas, and masks.
	dataset: The Dataset object to pick data from
	config: The model config object
	shuffle: If True, shuffles the samples before every epoch
	augment: (deprecated. Use augmentation instead). If true, apply random
		image augmentation. Currently, only horizontal flipping is offered.
	augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
		For example, passing imgaug.augmenters.Fliplr(0.5) flips images
		right/left 50% of the time.
	random_rois: If > 0 then generate proposals to be used to train the
				 network classifier and mask heads. Useful if training
				 the Mask RCNN part without the RPN.
	batch_size: How many images to return in each call
	detection_targets: If True, generate detection targets (class IDs, bbox
		deltas, and masks). Typically for debugging or visualizations because
		in trainig detection targets are generated by DetectionTargetLayer.
	no_augmentation_sources: Optional. List of sources to exclude for
		augmentation. A source is string that identifies a dataset and is
		defined in the Dataset class.
	Returns a Python generator. Upon calling next() on it, the
	generator returns two lists, inputs and outputs. The contents
	of the lists differs depending on the received arguments:
	inputs list:
	- images: [batch, H, W, C]
	- image_meta: [batch, (meta data)] Image details. See compose_image_meta()
	- rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
	- rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
	- gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
	- gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
	- gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
				are those of the image unless use_mini_mask is True, in which
				case they are defined in MINI_MASK_SHAPE.
	outputs list: Usually empty in regular training. But if detection_targets
		is True then the outputs list contains target class_ids, bbox deltas,
		and masks.
	"""
	b = 0  # batch item index
	image_index = -1
	image_ids = np.copy(dataset.image_ids)
	error_count = 0
	no_augmentation_sources = no_augmentation_sources or []

	# Anchors
	# [anchor_count, (y1, x1, y2, x2)]
	backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
	anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
											 config.RPN_ANCHOR_RATIOS,
											 backbone_shapes,
											 config.BACKBONE_STRIDES,
											 config.RPN_ANCHOR_STRIDE)

	# Keras requires a generator to run indefinitely.
	while True:
		try:
			# Increment index to pick next image. Shuffle if at the start of an epoch.
			image_index = (image_index + 1) % len(image_ids)
			if shuffle and image_index == 0:
				np.random.shuffle(image_ids)

			# Get GT bounding boxes and masks for image.
			image_id = image_ids[image_index]

			# If the image source is not to be augmented pass None as augmentation
			if not augmentation:
				#dataset.image_info[image_id]['source'] in no_augmentation_sources:
				image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
				load_image_gt2(dataset, config, image_id, augment=augment,
							  augmentation=None,
							  use_mini_mask=config.USE_MINI_MASK)
			else:
				image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
					load_image_gt2(dataset, config, image_id, augment=augment,
								augmentation=augmentation,
								use_mini_mask=config.USE_MINI_MASK)

			# Skip images that have no instances. This can happen in cases
			# where we train on a subset of classes and the image doesn't
			# have any of the classes we care about.
			if not np.any(gt_class_ids > 0):
				continue

			# RPN Targets
			rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors,
													gt_class_ids, gt_boxes, config)

			# Mask R-CNN Targets
			if random_rois:
				rpn_rois = generate_random_rois(
					image.shape, random_rois, gt_class_ids, gt_boxes)
				if detection_targets:
					rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask =\
						build_detection_targets(
							rpn_rois, gt_class_ids, gt_boxes, gt_masks, config)

			# Init batch arrays
			if b == 0:
				batch_image_meta = np.zeros(
					(batch_size,) + image_meta.shape, dtype=image_meta.dtype)
				batch_rpn_match = np.zeros(
					[batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
				batch_rpn_bbox = np.zeros(
					[batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
				batch_images = np.zeros(
					(batch_size,) + image.shape, dtype=np.float32)
				batch_gt_class_ids = np.zeros(
					(batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
				batch_gt_boxes = np.zeros(
					(batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)
				batch_gt_masks = np.zeros(
					(batch_size, gt_masks.shape[0], gt_masks.shape[1],
					 config.MAX_GT_INSTANCES), dtype=gt_masks.dtype)
				if random_rois:
					batch_rpn_rois = np.zeros(
						(batch_size, rpn_rois.shape[0], 4), dtype=rpn_rois.dtype)
					if detection_targets:
						batch_rois = np.zeros(
							(batch_size,) + rois.shape, dtype=rois.dtype)
						batch_mrcnn_class_ids = np.zeros(
							(batch_size,) + mrcnn_class_ids.shape, dtype=mrcnn_class_ids.dtype)
						batch_mrcnn_bbox = np.zeros(
							(batch_size,) + mrcnn_bbox.shape, dtype=mrcnn_bbox.dtype)
						batch_mrcnn_mask = np.zeros(
							(batch_size,) + mrcnn_mask.shape, dtype=mrcnn_mask.dtype)

			# If more instances than fits in the array, sub-sample from them.
			if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
				ids = np.random.choice(
					np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
				gt_class_ids = gt_class_ids[ids]
				gt_boxes = gt_boxes[ids]
				gt_masks = gt_masks[:, :, ids]

			# Add to batch
			batch_image_meta[b] = image_meta
			batch_rpn_match[b] = rpn_match[:, np.newaxis]
			batch_rpn_bbox[b] = rpn_bbox
			batch_images[b] = image#mold_image(image.astype(np.float32), config)
			batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
			batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
			batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
			if random_rois:
				batch_rpn_rois[b] = rpn_rois
				if detection_targets:
					batch_rois[b] = rois
					batch_mrcnn_class_ids[b] = mrcnn_class_ids
					batch_mrcnn_bbox[b] = mrcnn_bbox
					batch_mrcnn_mask[b] = mrcnn_mask

			b += 1

			# Batch full?
			if b >= batch_size:
				inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
						  batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
				outputs = []

				if random_rois:
					inputs.extend([batch_rpn_rois])
					if detection_targets:
						inputs.extend([batch_rois])
						# Keras requires that output and targets have the same number of dimensions
						batch_mrcnn_class_ids = np.expand_dims(
							batch_mrcnn_class_ids, -1)
						outputs.extend(
							[batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask])

				yield inputs, outputs

				# start a new batch
				b = 0
		except (GeneratorExit, KeyboardInterrupt):
			raise
		except:
			# Log it and skip the image
			image_id = image_ids[image_index]
			logging.exception("Error processing image {}".format(
				dataset.image_info[image_id]))
			error_count += 1
			if error_count > 5:
				raise


############################################################
#  MaskRCNN Class
############################################################

class MaskRCNN(object):
	"""Encapsulates the Mask RCNN model functionality.
	The actual Keras model is in the keras_model property.
	"""

	def __init__(self, mode, config, model_dir):
		"""
		mode: Either "training" or "inference"
		config: A Sub-class of the Config class
		model_dir: Directory to save training logs and trained weights
		"""
		assert mode in ['training', 'inference']
		self.mode = mode
		self.config = config
		self.model_dir = model_dir
		self.set_log_dir()
		self.keras_model = self.build(mode=mode, config=config)

	def build(self, mode, config):
		"""Build Mask R-CNN architecture.
			input_shape: The shape of the input image.
			mode: Either "training" or "inference". The inputs and
				outputs of the model differ accordingly.
		"""
		assert mode in ['training', 'inference']

		# Image size must be dividable by 2 multiple times
		h, w = config.IMAGE_SHAPE[:2]
		if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
			raise Exception("Image size must be dividable by 2 at least 6 times "
							"to avoid fractions when downscaling and upscaling."
							"For example, use 256, 320, 384, 448, 512, ... etc. ")

		# Inputs
		# input_image = KL.Input(
		#     shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
		# input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE],
		#                             name="input_image_meta")

		if config.INPUT_DIM == "3D":
			input_out = input_image = tf.keras.Input(
				shape=[None, None, config.INPUT_Z, 1], name="input_image")
			#this should remain the same.
			s = 1 + 4 + 4 + 4 + 1 + config.NUM_CLASSES
			input_image_meta = tf.keras.Input(shape=[s],
										name="input_image_meta")
		else:
		# Inputs
			input_out = input_image = tf.keras.Input(
				shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
			input_image_meta = tf.keras.Input(shape=[config.IMAGE_META_SIZE],
										name="input_image_meta")

		if mode == "training":
			# RPN GT
			input_rpn_match = KL.Input(
				shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
			input_rpn_bbox = KL.Input(
				shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

			# Detection GT (class IDs, bounding boxes, and masks)
			# 1. GT Class IDs (zero padded)
			input_gt_class_ids = KL.Input(
				shape=[None], name="input_gt_class_ids", dtype=tf.int32)
			# 2. GT Boxes in pixels (zero padded)
			# [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
			input_gt_boxes = KL.Input(
				shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
			# Normalize coordinates
			gt_boxes = KL.Lambda(lambda x: norm_boxes_graph(
				x, K.shape(input_image)[1:3]))(input_gt_boxes)
			# 3. GT Masks (zero padded)
			# [batch, height, width, MAX_GT_INSTANCES]
			if config.USE_MINI_MASK:
				input_gt_masks = KL.Input(
					shape=[config.MINI_MASK_SHAPE[0],
						   config.MINI_MASK_SHAPE[1], None],
					name="input_gt_masks", dtype=bool)
			else:
				input_gt_masks = KL.Input(
					shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
					name="input_gt_masks", dtype=bool)

		elif mode == "inference":
		   # Anchors in normalized coordinates
		   input_anchors = KL.Input(shape=[None, 4], name="input_anchors")

		# Build the shared convolutional layers.
		# Bottom-up Layers
		# Returns a list of the last layers of each stage, 5 in total.
		# Don't create the thead (stage 5), so we pick the 4th item in the list.
		# if callable(config.BACKBONE):
		#     _, C2, C3, C4, C5 = config.BACKBONE(input_image, stage5=True,
		#                                         train_bn=config.TRAIN_BN)
		# else:
		#     _, C2, C3, C4, C5 = resnet_graph(input_image, config.BACKBONE,
		#                                      stage5=True, train_bn=config.TRAIN_BN)

		if config.INPUT_DIM=="3D":
			###########
			### PEN ###
			###########
			stem_out = x = stem_graph_max(input_image, batch_size=config.BATCH_SIZE, z_slices=config.INPUT_Z, PEN_opts=config.PEN_opts, train_bn=config.TRAIN_BN)
			#produces a B x H x W x 3 output, typical input for mask-rcnn.
			if callable(config.BACKBONE):
				_, C2, C3, C4, C5 = config.BACKBONE(x, stage5=True,
													train_bn=config.TRAIN_BN)
			else:
				#print("here")
				_, C2, C3, C4, C5 = resnet_graph(x, config.BACKBONE,
												 stage5=True, train_bn=config.TRAIN_BN)

		else:
			if callable(config.BACKBONE):
				_, C2, C3, C4, C5 = config.BACKBONE(input_image, stage5=True,
													train_bn=config.TRAIN_BN)
			else:
				_, C2, C3, C4, C5 = resnet_graph(input_image, config.BACKBONE,
												 stage5=True, train_bn=config.TRAIN_BN)
		# Top-down Layers
		# TODO: add assert to varify feature map sizes match what's in config
		P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
		P4 = KL.Add(name="fpn_p4add")([
			KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
			KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
		P3 = KL.Add(name="fpn_p3add")([
			KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
			KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
		P2 = KL.Add(name="fpn_p2add")([
			KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
			KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])
		# Attach 3x3 conv to all P layers to get the final feature maps.
		P2 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
		P3 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
		P4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
		P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)
		# P6 is used for the 5th anchor scale in RPN. Generated by
		# subsampling from P5 with stride of 2.
		P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

		# Note that P6 is used in RPN, but not in the classifier heads.
		rpn_feature_maps = [P2, P3, P4, P5, P6]
		mrcnn_feature_maps = [P2, P3, P4, P5]

		# Anchors
		if mode == "training":
			anchors = self.get_anchors(config.IMAGE_SHAPE)
			# Duplicate across the batch dimension because Keras requires it
			# TODO: can this be optimized to avoid duplicating the anchors?
			anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
			# A hack to get around Keras's bad support for constants
			#anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
			if config.INPUT_DIM=="3D":
				#anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(x)
				anchors = AnchorsLayer(anchors, name="anchors")(x)
			else:
				#anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
				anchors = AnchorsLayer(anchors, name="anchors")(input_image)
		else:
			#anchors = input_anchors
			anchors = self.get_anchors(config.IMAGE_SHAPE)
			# Duplicate across the batch dimension because Keras requires it
			# TODO: can this be optimized to avoid duplicating the anchors?
			anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
			# A hack to get around Keras's bad support for constants
			#anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
			if config.INPUT_DIM=="3D":
				#anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(x)
				anchors = AnchorsLayer(anchors, name="anchors_inference")(x)
			else:
				#anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
				anchors = AnchorsLayer(anchors, name="anchors_inference")(input_image)


		#print(anchors.shape)

		# RPN Model
		rpn = build_rpn_model(config.RPN_ANCHOR_STRIDE,
							  len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)
		# Loop through pyramid layers
		layer_outputs = []  # list of lists
		for p in rpn_feature_maps:
			layer_outputs.append(rpn([p]))
		# Concatenate layer outputs
		# Convert from list of lists of level outputs to list of lists
		# of outputs across levels.
		# e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
		output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
		outputs = list(zip(*layer_outputs))
		outputs = [KL.Concatenate(axis=1, name=n)(list(o))
				   for o, n in zip(outputs, output_names)]

		rpn_class_logits, rpn_class, rpn_bbox = outputs

		# Generate proposals
		# Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
		# and zero padded.
		proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training"\
			else config.POST_NMS_ROIS_INFERENCE
		rpn_rois = ProposalLayer(
			proposal_count=proposal_count,
			nms_threshold=config.RPN_NMS_THRESHOLD,
			name="ROI",
			config=config)([rpn_class, rpn_bbox, anchors])

		if mode == "training":
			# Class ID mask to mark class IDs supported by the dataset the image
			# came from.
			active_class_ids = KL.Lambda(
				lambda x: parse_image_meta_graph(x)["active_class_ids"]
				)(input_image_meta)

			if not config.USE_RPN_ROIS:
				# Ignore predicted ROIs and use ROIs provided as an input.
				input_rois = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
									  name="input_roi", dtype=np.int32)
				# Normalize coordinates
				target_rois = KL.Lambda(lambda x: norm_boxes_graph(
					x, K.shape(input_image)[1:3]))(input_rois)
			else:
				target_rois = rpn_rois

			# Generate detection targets
			# Subsamples proposals and generates target outputs for training
			# Note that proposal class IDs, gt_boxes, and gt_masks are zero
			# padded. Equally, returned rois and targets are zero padded.
			rois, target_class_ids, target_bbox, target_mask =\
				DetectionTargetLayer(config, name="proposal_targets")([
					target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

			# Network Heads
			# TODO: verify that this handles zero padded ROIs
			mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
				fpn_classifier_graph(rois, mrcnn_feature_maps, input_image_meta,
									 config.POOL_SIZE, config.NUM_CLASSES,
									 train_bn=config.TRAIN_BN,
									 fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

			mrcnn_mask = build_fpn_mask_graph(rois, mrcnn_feature_maps,
											  input_image_meta,
											  config.MASK_POOL_SIZE,
											  config.NUM_CLASSES,
											  train_bn=config.TRAIN_BN)

			# TODO: clean up (use tf.identify if necessary)
			output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

			# Losses
			rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
				[input_rpn_match, rpn_class_logits])
			rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
				[input_rpn_bbox, input_rpn_match, rpn_bbox])
			class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
				[target_class_ids, mrcnn_class_logits, active_class_ids])
			bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
				[target_bbox, target_class_ids, mrcnn_bbox])
			mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
				[target_mask, target_class_ids, mrcnn_mask])

			# Model
			inputs = [input_image, input_image_meta,
					  input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks]
			if not config.USE_RPN_ROIS:
				inputs.append(input_rois)

			if not config.Testing:
				outputs = [rpn_class_logits, rpn_class, rpn_bbox,
						   mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
						   rpn_rois, output_rois,
						   rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
			else:
				outputs = [rpn_class_logits, rpn_class, rpn_bbox,
						   mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
						   rpn_rois, output_rois,
						   rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss, input_out]

			model = KM.Model(inputs, outputs, name='mask_rcnn')
		else:
			# Network Heads
			# Proposal classifier and BBox regressor heads
			mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
				fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
									 config.POOL_SIZE, config.NUM_CLASSES,
									 train_bn=config.TRAIN_BN,
									 fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

			# Detections
			# output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in
			# normalized coordinates
			detections = DetectionLayer(config, name="mrcnn_detection")(
				[rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

			# Create masks for detections
			detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
			mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
											  input_image_meta,
											  config.MASK_POOL_SIZE,
											  config.NUM_CLASSES,
											  train_bn=config.TRAIN_BN)

			if config.Testing:
				if config.INPUT_DIM=="3D":
					model = KM.Model([input_image, input_image_meta, input_anchors],
									 [detections, mrcnn_class, mrcnn_bbox,
										 mrcnn_mask, rpn_rois, rpn_class, rpn_bbox, stem_out, input_out],
									 name='mask_rcnn')
				else:
					model = KM.Model([input_image, input_image_meta, input_anchors],
									 [detections, mrcnn_class, mrcnn_bbox,
										 mrcnn_mask, rpn_rois, rpn_class, rpn_bbox, input_out],
									 name='mask_rcnn')
			else:
				model = KM.Model([input_image, input_image_meta, input_anchors],
								 [detections, mrcnn_class, mrcnn_bbox,
									 mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
								 name='mask_rcnn')

		# Add multi-GPU support.
		if config.GPU_COUNT > 1:
			from mrcnn.parallel_model import ParallelModel
			model = ParallelModel(model, config.GPU_COUNT)

		return model

	def find_last(self):
		"""Finds the last checkpoint file of the last trained model in the
		model directory.
		Returns:
			The path of the last checkpoint file
		"""
		# Get directory names. Each directory corresponds to a model
		dir_names = next(os.walk(self.model_dir))[1]
		key = self.config.NAME.lower()
		dir_names = filter(lambda f: f.startswith(key), dir_names)
		dir_names = sorted(dir_names)
		if not dir_names:
			import errno
			raise FileNotFoundError(
				errno.ENOENT,
				"Could not find model directory under {}".format(self.model_dir))
		# Pick last directory
		dir_name = os.path.join(self.model_dir, dir_names[-1])
		# Find the last checkpoint
		checkpoints = next(os.walk(dir_name))[2]
		checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
		checkpoints = sorted(checkpoints)
		if not checkpoints:
			import errno
			raise FileNotFoundError(
				errno.ENOENT, "Could not find weight files in {}".format(dir_name))
		checkpoint = os.path.join(dir_name, checkpoints[-1])
		return checkpoint

	def load_weights(self, filepath, by_name=False, exclude=None):
		"""Modified version of the corresponding Keras function with
		the addition of multi-GPU support and the ability to exclude
		some layers from loading.
		exclude: list of layer names to exclude
		"""
		import h5py
		from tensorflow.python.keras.saving import hdf5_format

		if exclude:
			by_name = True

		if h5py is None:
			raise ImportError('`load_weights` requires h5py.')
		with h5py.File(filepath, mode='r') as f:
			if 'layer_names' not in f.attrs and 'model_weights' in f:
				f = f['model_weights']

			# In multi-GPU training, we wrap the model. Get layers
			# of the inner model because they have the weights.
			keras_model = self.keras_model
			layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
				else keras_model.layers

			# Exclude some layers
			if exclude:
				layers = filter(lambda l: l.name not in exclude, layers)

			if by_name:
				hdf5_format.load_weights_from_hdf5_group_by_name(f, layers)
			else:
				hdf5_format.load_weights_from_hdf5_group(f, layers)

		# Update the log directory
		self.set_log_dir(filepath)

	def get_imagenet_weights(self):
		"""Downloads ImageNet trained weights from Keras.
		Returns path to weights file.
		"""
		from keras.utils.data_utils import get_file
		TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/'\
								 'releases/download/v0.2/'\
								 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
		weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
								TF_WEIGHTS_PATH_NO_TOP,
								cache_subdir='models',
								md5_hash='a268eb855778b3df3c7506639542a6af')
		return weights_path

	def compile(self, learning_rate, momentum):
		"""Gets the model ready for training. Adds losses, regularization, and
		metrics. Then calls the Keras compile() function.
		"""
		# Optimizer object
		optimizer = keras.optimizers.SGD(
			lr=learning_rate, momentum=momentum,
			clipnorm=self.config.GRADIENT_CLIP_NORM)
		# Add Losses
		loss_names = [
			"rpn_class_loss",  "rpn_bbox_loss",
			"mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]
		for name in loss_names:
			layer = self.keras_model.get_layer(name)
			if layer.output in self.keras_model.losses:
				continue
			loss = (
				tf.reduce_mean(input_tensor=layer.output, keepdims=True)
				* self.config.LOSS_WEIGHTS.get(name, 1.))
			self.keras_model.add_loss(loss)

		# Add L2 Regularization
		# Skip gamma and beta weights of batch normalization layers.
		reg_losses = [
			keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(input=w), tf.float32)
			for w in self.keras_model.trainable_weights
			if 'gamma' not in w.name and 'beta' not in w.name]
		self.keras_model.add_loss(tf.add_n(reg_losses))

		# Compile
		self.keras_model.compile(
			optimizer=optimizer,
			loss=[None] * len(self.keras_model.outputs))

		# Add metrics for losses
		for name in loss_names:
			if name in self.keras_model.metrics_names:
				continue
			layer = self.keras_model.get_layer(name)
			self.keras_model.metrics_names.append(name)
			loss = (
				tf.reduce_mean(input_tensor=layer.output, keepdims=True)
				* self.config.LOSS_WEIGHTS.get(name, 1.))
			self.keras_model.add_metric(loss, name=name, aggregation='mean')

	def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
		"""Sets model layers as trainable if their names match
		the given regular expression.
		"""
		# Print message on the first call (but not on recursive calls)
		if verbose > 0 and keras_model is None:
			log("Selecting layers to train")

		keras_model = keras_model or self.keras_model

		# In multi-GPU training, we wrap the model. Get layers
		# of the inner model because they have the weights.
		layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
			else keras_model.layers

		for layer in layers:
			# Is the layer a model?
			if layer.__class__.__name__ == 'Model':
				print("In model: ", layer.name)
				self.set_trainable(
					layer_regex, keras_model=layer, indent=indent + 4)
				continue

			if not layer.weights:
				continue
			# Is it trainable?
			trainable = bool(re.fullmatch(layer_regex, layer.name))
			# Update layer. If layer is a container, update inner layer.
			if layer.__class__.__name__ == 'TimeDistributed':
				layer.layer.trainable = trainable
			else:
				layer.trainable = trainable
			# Print trainable layer names
			if trainable and verbose > 0:
				log("{}{:20}   ({})".format(" " * indent, layer.name,
											layer.__class__.__name__))

	def set_log_dir(self, model_path=None):
		"""Sets the model log directory and epoch counter.
		model_path: If None, or a format different from what this code uses
			then set a new log directory and start epochs from 0. Otherwise,
			extract the log directory and the epoch counter from the file
			name.
		"""
		# Set date and epoch counter as if starting a new model
		self.epoch = 0
		now = datetime.datetime.now()

		# If we have a model path with date and epochs use them
		if model_path:
			# Continue from we left of. Get epoch and date from the file name
			# A sample model path might look like:
			# \path\to\logs\coco20171029T2315\mask_rcnn_coco_0001.h5 (Windows)
			# /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5 (Linux)
			regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]mask\_rcnn\_[\w-]+(\d{4})\.h5"
			# Use string for regex since we might want to use pathlib.Path as model_path
			m = re.match(regex, str(model_path))
			if m:
				now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
										int(m.group(4)), int(m.group(5)))
				# Epoch number in file is 1-based, and in Keras code it's 0-based.
				# So, adjust for that then increment by one to start from the next epoch
				self.epoch = int(m.group(6)) - 1 + 1
				print('Re-starting from epoch %d' % self.epoch)

		# Directory for training logs
		self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
			self.config.NAME.lower(), now))

		# Path to save after each epoch. Include placeholders that get filled by Keras.
		self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.h5".format(
			self.config.NAME.lower()))
		self.checkpoint_path = self.checkpoint_path.replace(
			"*epoch*", "{epoch:04d}")
		#self.model_path = os.path.join(self.log_dir, "mask_rcnn_model_{}.h5".format(self.config.INPUT_DIM))

	def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
			  augmentation=None, custom_callbacks=None, no_augmentation_sources=None):
		"""Train the model.
		train_dataset, val_dataset: Training and validation Dataset objects.
		learning_rate: The learning rate to train with
		epochs: Number of training epochs. Note that previous training epochs
				are considered to be done alreay, so this actually determines
				the epochs to train in total rather than in this particaular
				call.
		layers: Allows selecting wich layers to train. It can be:
			- A regular expression to match layer names to train
			- One of these predefined values:
			  heads: The RPN, classifier and mask heads of the network
			  all: All the layers
			  stem: Train all Stem Layers
			  3+: Train Resnet stage 3 and up
			  4+: Train Resnet stage 4 and up
			  5+: Train Resnet stage 5 and up
		augmentation: Optional. An imgaug (https://github.com/aleju/imgaug)
			augmentation. For example, passing imgaug.augmenters.Fliplr(0.5)
			flips images right/left 50% of the time. You can pass complex
			augmentations as well. This augmentation applies 50% of the
			time, and when it does it flips images right/left half the time
			and adds a Gaussian blur with a random sigma in range 0 to 5.
				augmentation = imgaug.augmenters.Sometimes(0.5, [
					imgaug.augmenters.Fliplr(0.5),
					imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
				])
		custom_callbacks: Optional. Add custom callbacks to be called
			with the keras fit_generator method. Must be list of type keras.callbacks.
		no_augmentation_sources: Optional. List of sources to exclude for
			augmentation. A source is string that identifies a dataset and is
			defined in the Dataset class.
		"""
		assert self.mode == "training", "Create model in training mode."

		# Pre-defined layer regular expressions
		layer_regex = {
			# all layers but the backbone
			"heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
			# From a specific Resnet stage and up
			"3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
			"4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
			"5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
			# All layers
			"all": ".*",
			#Just mask rcnn layers
			"mask_rcnn": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(res5.*)|(bn5.*)",
			#only train PEN module layers.
			"stem": r"(stem_conv.*)|(stem_bn.*)|(stem_act.*)",
		}
		if layers in layer_regex.keys():
			layers = layer_regex[layers]

		# Data generators
		train_generator = data_generator(train_dataset, self.config, shuffle=True,
										 augmentation=augmentation,
										 batch_size=self.config.BATCH_SIZE,
										 no_augmentation_sources=no_augmentation_sources)

		def_setting = self.config.IMAGE_RESIZE_MODE
		self.config.IMAGE_RESIZE_MODE = "center"
		val_generator = data_generator(val_dataset, self.config, shuffle=True,
									   augmentation=augmentation,
									   batch_size=self.config.BATCH_SIZE)
		self.config.IMAGE_RESIZE_MODE = def_setting
		#CE Changed 11/28/21 for reproducibility of validation set.

		# Create log_dir if it does not exist
		if not os.path.exists(self.log_dir):
			os.makedirs(self.log_dir)

		# Callbacks
		callbacks = [
			keras.callbacks.TensorBoard(log_dir=self.log_dir,
										histogram_freq=0, write_graph=True, write_images=False),
			keras.callbacks.ModelCheckpoint(self.checkpoint_path,
											verbose=0, save_weights_only=True, save_best_only=True,
											monitor='val_loss',mode='min'),
		]

		# Add custom callbacks to the list
		if custom_callbacks:
			callbacks += custom_callbacks

		# Train
		log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
		log("Checkpoint Path: {}".format(self.checkpoint_path))
		self.set_trainable(layers)
		self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

		# Work-around for Windows: Keras fails on Windows when using
		# multiprocessing workers. See discussion here:
		# https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
		if os.name == 'nt':
			workers = 0
		else:
			workers = multiprocessing.cpu_count()

		# XXX make multiprocessing work again
		workers = 1

		self.keras_model.fit(
			train_generator,
			initial_epoch=self.epoch,
			epochs=epochs,
			steps_per_epoch=self.config.STEPS_PER_EPOCH,
			callbacks=callbacks,
			validation_data=val_generator,
			validation_steps=self.config.VALIDATION_STEPS,
			max_queue_size=100,
			workers=workers,
			use_multiprocessing=workers > 1,
		)
		self.epoch = max(self.epoch, epochs)

	def mold_inputs(self, images):
		"""Takes a list of images and modifies them to the format expected
		as an input to the neural network.
		images: List of image matrices [height,width,depth]. Images can have
			different sizes.
		Returns 3 Numpy matrices:
		molded_images: [N, h, w, 3]. Images resized and normalized.
		image_metas: [N, length of meta data]. Details about each image.
		windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
			original image (padding excluded).
		"""
		molded_images = []
		image_metas = []
		windows = []
		for image in images:
			# Resize image
			# TODO: move resizing to mold_image()
			#print(image.shape)
			molded_image, window, scale, padding, crop = utils.resize_image(
				image,
				min_dim=self.config.IMAGE_MIN_DIM,
				min_scale=self.config.IMAGE_MIN_SCALE,
				max_dim=self.config.IMAGE_MAX_DIM,
				mode=self.config.IMAGE_RESIZE_MODE)
			#print(molded_image.shape)
			#molded_image = mold_image(molded_image, self.config)
			# Build image_meta
			image_meta = compose_image_meta(
				0, image.shape, molded_image.shape, window, scale,
				np.ones([self.config.NUM_CLASSES], dtype=np.int32))
			# Append
			molded_images.append(molded_image)
			windows.append(window)
			image_metas.append(image_meta)
		# Pack into arrays
		molded_images = np.stack(molded_images)
		image_metas = np.stack(image_metas)
		windows = np.stack(windows)
		return molded_images, image_metas, windows

	def unmold_detections(self, detections, mrcnn_mask, original_image_shape,
						  image_shape, window):
		"""Reformats the detections of one image from the format of the neural
		network output to a format suitable for use in the rest of the
		application.
		detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
		mrcnn_mask: [N, height, width, num_classes]
		original_image_shape: [H, W, C] Original image shape before resizing
		Changed CE: Can be [H,W,Z,C]
		image_shape: [H, W, C] Shape of the image after resizing and padding
		window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
				image is excluding the padding.
		Returns:
		boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
		class_ids: [N] Integer class IDs for each bounding box
		scores: [N] Float probability scores of the class_id
		masks: [height, width, num_instances] Instance masks
		"""
		# How many detections do we have?
		# Detections array is padded with zeros. Find the first class_id == 0.
		zero_ix = np.where(detections[:, 4] == 0)[0]
		N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

		# Extract boxes, class_ids, scores, and class-specific masks
		boxes = detections[:N, :4]
		class_ids = detections[:N, 4].astype(np.int32)
		scores = detections[:N, 5]
		masks = mrcnn_mask[np.arange(N), :, :, class_ids]

		# Translate normalized coordinates in the resized image to pixel
		# coordinates in the original image before resizing
		window = utils.norm_boxes(window, image_shape[:2])
		wy1, wx1, wy2, wx2 = window
		shift = np.array([wy1, wx1, wy1, wx1])
		wh = wy2 - wy1  # window height
		ww = wx2 - wx1  # window width
		scale = np.array([wh, ww, wh, ww])
		# Convert boxes to normalized coordinates on the window
		boxes = np.divide(boxes - shift, scale)
		# Convert boxes to pixel coordinates on the original image
		boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

		# Filter out detections with zero area. Happens in early training when
		# network weights are still random
		exclude_ix = np.where(
			(boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
		if exclude_ix.shape[0] > 0:
			boxes = np.delete(boxes, exclude_ix, axis=0)
			class_ids = np.delete(class_ids, exclude_ix, axis=0)
			scores = np.delete(scores, exclude_ix, axis=0)
			masks = np.delete(masks, exclude_ix, axis=0)
			N = class_ids.shape[0]

		# Resize masks to original image size and set boundary threshold.
		full_masks = []
		for i in range(N):
			# Convert neural network mask to full size mask
			full_mask = utils.unmold_mask(masks[i], boxes[i], (*original_image_shape[:2],3))
			full_masks.append(full_mask)
		full_masks = np.stack(full_masks, axis=-1)\
			if full_masks else np.empty(original_image_shape[:2] + (0,))

		return boxes, class_ids, scores, full_masks

	def detect(self, images, verbose=0):
		"""Runs the detection pipeline.
		images: List of images, potentially of different sizes.
		Returns a list of dicts, one dict per image. The dict contains:
		rois: [N, (y1, x1, y2, x2)] detection bounding boxes
		class_ids: [N] int class IDs
		scores: [N] float probability scores for the class IDs
		masks: [H, W, N] instance binary masks
		"""
		assert self.mode == "inference", "Create model in inference mode."
		print(len(images))
		assert len(images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

		if verbose:
			log("Processing {} images".format(len(images)))
			for image in images:
				log("image", image)
		# Mold inputs to format expected by the neural network
		molded_images, image_metas, windows = self.mold_inputs(images)
		#we need image metas to look correct.
		#also, below in image_shape, we likely need that to be the (*molded_images[0][:2],3)? Should be 3 channel...

		# Validate image sizes
		# All images in a batch MUST be of the same size
		image_shape = molded_images[0].shape
		for g in molded_images[1:]:
			assert g.shape == image_shape,\
				"After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

		# Anchors
		anchors = self.get_anchors(np.array(list(image_shape[:2]+(self.config.IMAGE_CHANNEL_COUNT,))))
		#image_shape)
		# Duplicate across the batch dimension because Keras requires it
		# TODO: can this be optimized to avoid duplicating the anchors?
		anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

		if verbose:
			log("molded_images", molded_images)
			log("image_metas", image_metas)
			log("anchors", anchors)
		# Run object detection
		# a = tf.constant([[1, 2], [3, 4]])
		# if self.config.INPUT_DIM=="3D":
		#     detections_call, _, _, mrcnn_mask_call, _, _, _, stem_out_call, input_image =\
		#         self.keras_model([molded_images, image_metas, anchors], training=False)#, verbose=0)
		#     in_max = input_image.eval(session=tf.compat.v1.Session())
		#     in_max = in_max[0]
		#     in_max = np.max(in_max,axis=2)
		#     #okay, so the images are good now.
		# else:
		#     detections_call, _, _, mrcnn_mask_call, _, _, _, input_image_call=\
		#         self.keras_model([molded_images, image_metas, anchors], training=False)#, verbose=0)
		# detections, _, _, mrcnn_mask, _, _, _, stem_out =\
		#     self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
		detections, _, _, mrcnn_mask, _, _, _=\
			self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
		#the above segfaults for 3D input
		# Process detections
		#out.eval(session=tf.compat.v1.Session())
		#a = a.eval(session=tf.compat.v1.Session()) #works.
		#the following error out.
		#stem_out = stem_out.eval(session=tf.compat.v1.Session())
		# detections = detections_call.eval(session=tf.compat.v1.Session())
		# mrcnn_mask = mrcnn_mask_call.eval(session=tf.compat.v1.Session())
		results = []
		for i, image in enumerate(images):
			final_rois, final_class_ids, final_scores, final_masks =\
				self.unmold_detections(detections[i], mrcnn_mask[i],
									   image.shape, molded_images[i].shape,
									   windows[i])
			results.append({
				"rois": final_rois,
				"class_ids": final_class_ids,
				"scores": final_scores,
				"masks": final_masks,
			})
		return [results, molded_images]

	def detect_molded(self, molded_images, image_metas, verbose=0):
		"""Runs the detection pipeline, but expect inputs that are
		molded already. Used mostly for debugging and inspecting
		the model.
		molded_images: List of images loaded using load_image_gt()
		image_metas: image meta data, also returned by load_image_gt()
		Returns a list of dicts, one dict per image. The dict contains:
		rois: [N, (y1, x1, y2, x2)] detection bounding boxes
		class_ids: [N] int class IDs
		scores: [N] float probability scores for the class IDs
		masks: [H, W, N] instance binary masks
		"""
		assert self.mode == "inference", "Create model in inference mode."
		assert len(molded_images) == self.config.BATCH_SIZE,\
			"Number of images must be equal to BATCH_SIZE"

		if verbose:
			log("Processing {} images".format(len(molded_images)))
			for image in molded_images:
				log("image", image)

		# Validate image sizes
		# All images in a batch MUST be of the same size
		image_shape = molded_images[0].shape
		for g in molded_images[1:]:
			assert g.shape == image_shape, "Images must have the same size"

		# Anchors
		anchors = self.get_anchors(image_shape)
		# Duplicate across the batch dimension because Keras requires it
		# TODO: can this be optimized to avoid duplicating the anchors?
		anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

		if verbose:
			log("molded_images", molded_images)
			log("image_metas", image_metas)
			log("anchors", anchors)
		# Run object detection
		detections, _, _, mrcnn_mask, _, _, _ =\
			self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
		# Process detections
		results = []
		for i, image in enumerate(molded_images):
			window = [0, 0, image.shape[0], image.shape[1]]
			final_rois, final_class_ids, final_scores, final_masks =\
				self.unmold_detections(detections[i], mrcnn_mask[i],
									   image.shape, molded_images[i].shape,
									   window)
			results.append({
				"rois": final_rois,
				"class_ids": final_class_ids,
				"scores": final_scores,
				"masks": final_masks,
			})
		return results

	def get_anchors(self, image_shape):
		"""Returns anchor pyramid for the given image size."""
		backbone_shapes = compute_backbone_shapes(self.config, image_shape)
		# Cache anchors and reuse if image shape is the same
		if not hasattr(self, "_anchor_cache"):
			self._anchor_cache = {}
		if not tuple(image_shape) in self._anchor_cache:
			# Generate Anchors
			a = utils.generate_pyramid_anchors(
				self.config.RPN_ANCHOR_SCALES,
				self.config.RPN_ANCHOR_RATIOS,
				backbone_shapes,
				self.config.BACKBONE_STRIDES,
				self.config.RPN_ANCHOR_STRIDE)
			# Keep a copy of the latest anchors in pixel coordinates because
			# it's used in inspect_model notebooks.
			# TODO: Remove this after the notebook are refactored to not use it
			self.anchors = a
			# Normalize coordinates
			self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(a, image_shape[:2])
		return self._anchor_cache[tuple(image_shape)]

	def ancestor(self, tensor, name, checked=None):
		"""Finds the ancestor of a TF tensor in the computation graph.
		tensor: TensorFlow symbolic tensor.
		name: Name of ancestor tensor to find
		checked: For internal use. A list of tensors that were already
				 searched to avoid loops in traversing the graph.
		"""
		checked = checked if checked is not None else []
		# Put a limit on how deep we go to avoid very long loops
		if len(checked) > 500:
			return None
		# Convert name to a regex and allow matching a number prefix
		# because Keras adds them automatically
		if isinstance(name, str):
			name = re.compile(name.replace("/", r"(\_\d+)*/"))

		parents = tensor.op.inputs
		for p in parents:
			if p in checked:
				continue
			if bool(re.fullmatch(name, p.name)):
				return p
			checked.append(p)
			a = self.ancestor(p, name, checked)
			if a is not None:
				return a
		return None

	def find_trainable_layer(self, layer):
		"""If a layer is encapsulated by another layer, this function
		digs through the encapsulation and returns the layer that holds
		the weights.
		"""
		if layer.__class__.__name__ == 'TimeDistributed':
			return self.find_trainable_layer(layer.layer)
		return layer

	def get_trainable_layers(self):
		"""Returns a list of layers that have weights."""
		layers = []
		# Loop through all layers
		for l in self.keras_model.layers:
			# If layer is a wrapper, find inner trainable layer
			l = self.find_trainable_layer(l)
			# Include layer if it has weights
			if l.get_weights():
				layers.append(l)
		return layers

	def run_graph(self, images, outputs, image_metas=None):
		"""Runs a sub-set of the computation graph that computes the given
		outputs.
		image_metas: If provided, the images are assumed to be already
			molded (i.e. resized, padded, and normalized)
		outputs: List of tuples (name, tensor) to compute. The tensors are
			symbolic TensorFlow tensors and the names are for easy tracking.
		Returns an ordered dict of results. Keys are the names received in the
		input and values are Numpy arrays.
		"""
		model = self.keras_model

		# Organize desired outputs into an ordered dict
		outputs = OrderedDict(outputs)
		for o in outputs.values():
			assert o is not None

		# Build a Keras function to run parts of the computation graph
		inputs = model.inputs
		if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
			inputs += [K.learning_phase()]
		kf = K.function(model.inputs, list(outputs.values()))

		# Prepare inputs
		if image_metas is None:
			molded_images, image_metas, _ = self.mold_inputs(images)
		else:
			molded_images = images
		image_shape = molded_images[0].shape
		# Anchors
		anchors = self.get_anchors(image_shape)
		# Duplicate across the batch dimension because Keras requires it
		# TODO: can this be optimized to avoid duplicating the anchors?
		anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
		model_in = [molded_images, image_metas, anchors]

		# Run inference
		if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
			model_in.append(0.)
		outputs_np = kf(model_in)

		# Pack the generated Numpy arrays into a a dict and log the results.
		outputs_np = OrderedDict([(k, v)
								  for k, v in zip(outputs.keys(), outputs_np)])
		for k, v in outputs_np.items():
			log(k, v)
		return outputs_np


############################################################
#  Data Formatting
############################################################

def compose_image_meta(image_id, original_image_shape, image_shape,
					   window, scale, active_class_ids):
	"""Takes attributes of an image and puts them in one 1D array.
	image_id: An int ID of the image. Useful for debugging.
	original_image_shape: [H, W, C] before resizing or padding.
	image_shape: [H, W, C] after resizing and padding
	window: (y1, x1, y2, x2) in pixels. The area of the image where the real
			image is (excluding the padding)
	scale: The scaling factor applied to the original image (float32)
	active_class_ids: List of class_ids available in the dataset from which
		the image came. Useful if training on images from multiple datasets
		where not all classes are present in all datasets.
	"""
	meta = np.array(
		[image_id] +                  # size=1
		list(original_image_shape) +  # size=3
		list(image_shape) +           # size=3
		list(window) +                # size=4 (y1, x1, y2, x2) in image cooredinates
		[scale] +                     # size=1
		list(active_class_ids)        # size=num_classes
	)
	return meta


def parse_image_meta(meta):
	"""Parses an array that contains image attributes to its components.
	See compose_image_meta() for more details.
	meta: [batch, meta length] where meta length depends on NUM_CLASSES
	Returns a dict of the parsed values.
	"""
	image_id = meta[:, 0]
	original_image_shape = meta[:, 1:4]
	image_shape = meta[:, 4:7]
	window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
	scale = meta[:, 11]
	active_class_ids = meta[:, 12:]
	return {
		"image_id": image_id.astype(np.int32),
		"original_image_shape": original_image_shape.astype(np.int32),
		"image_shape": image_shape.astype(np.int32),
		"window": window.astype(np.int32),
		"scale": scale.astype(np.float32),
		"active_class_ids": active_class_ids.astype(np.int32),
	}


def parse_image_meta_graph(meta):
	"""Parses a tensor that contains image attributes to its components.
	See compose_image_meta() for more details.
	meta: [batch, meta length] where meta length depends on NUM_CLASSES
	Returns a dict of the parsed tensors.
	"""
	image_id = meta[:, 0]
	original_image_shape = meta[:, 1:4]
	image_shape = meta[:, 4:7]
	window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
	scale = meta[:, 11]
	active_class_ids = meta[:, 12:]
	return {
		"image_id": image_id,
		"original_image_shape": original_image_shape,
		"image_shape": image_shape,
		"window": window,
		"scale": scale,
		"active_class_ids": active_class_ids,
	}


def mold_image(images, config):
	print("im_shape=", images.shape)
	"""Expects an RGB image (or array of images) and subtracts
	the mean pixel and converts it to float. Expects image
	colors in RGB order.
	"""
	return images.astype(np.float32) - config.MEAN_PIXEL


def unmold_image(normalized_images, config):
	"""Takes a image normalized with mold() and returns the original."""
	return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)


############################################################
#  Miscellenous Graph Functions
############################################################

def trim_zeros_graph(boxes, name='trim_zeros'):
	"""Often boxes are represented with matrices of shape [N, 4] and
	are padded with zeros. This removes zero boxes.
	boxes: [N, 4] matrix of boxes.
	non_zeros: [N] a 1D boolean mask identifying the rows to keep
	"""
	non_zeros = tf.cast(tf.reduce_sum(input_tensor=tf.abs(boxes), axis=1), tf.bool)
	boxes = tf.boolean_mask(tensor=boxes, mask=non_zeros, name=name)
	return boxes, non_zeros


def batch_pack_graph(x, counts, num_rows):
	"""Picks different number of values from each row
	in x depending on the values in counts.
	"""
	outputs = []
	for i in range(num_rows):
		outputs.append(x[i, :counts[i]])
	return tf.concat(outputs, axis=0)


def norm_boxes_graph(boxes, shape):
	"""Converts boxes from pixel coordinates to normalized coordinates.
	boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
	shape: [..., (height, width)] in pixels
	Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
	coordinates it's inside the box.
	Returns:
		[..., (y1, x1, y2, x2)] in normalized coordinates
	"""
	h, w = tf.split(tf.cast(shape, tf.float32), 2)
	scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
	shift = tf.constant([0., 0., 1., 1.])
	return tf.divide(boxes - shift, scale)


def denorm_boxes_graph(boxes, shape):
	"""Converts boxes from normalized coordinates to pixel coordinates.
	boxes: [..., (y1, x1, y2, x2)] in normalized coordinates
	shape: [..., (height, width)] in pixels
	Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
	coordinates it's inside the box.
	Returns:
		[..., (y1, x1, y2, x2)] in pixel coordinates
	"""
	h, w = tf.split(tf.cast(shape, tf.float32), 2)
	scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
	shift = tf.constant([0., 0., 1., 1.])
	return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32)

############################################################################
############################ AUGMENTATIONS #################################
############################################################################

def image_mask_rotate(image, mask, centroids, prob):
	#Rotation direction is from the first towards the second axis.
	#so rotates data counter-clockwise
	#if the image is H x W, and a centroid is at h,w when we rotate with rot90,
	#the new centroid becomes W-w, h
	#if we do a flipud,
	#centroids should be a
	if np.random.rand()>=prob:
		image = np.rot90(image)
		imshape = image.shape[:2]
		mask = np.rot90(mask)
		centroids_updated = copy.deepcopy(centroids)
		centroids_updated[:,0] = imshape[1]-centroids[:,1]
		centroids_updated[:,1] = centroids[:,0]
		flag=True
		return image, mask, centroids_updated, True
	else:
		return image, mask, centroids, False

def image_xyflip(image, mask, centroids, prob):
	imshape = image.shape[:2]
	flag=False
	if np.random.rand()>=prob:
		image = image[::-1,:]#image[::-1,:,:,:]
		mask = mask[::-1,:,:]
		centroids[:,0] = imshape[0] - centroids[:,0]
		flag=True
	if np.random.rand()>=prob:
		image = image[:,::-1]#image[:,::-1,:,:]
		mask = mask[:,::-1,:]
		centroids[:,1] = imshape[1] - centroids[:,1]
		flag=True
	return image, mask, centroids, flag

def image_zflip(image, im_z_size, zlocs, prob):
	flag = False
	if np.random.rand()>=prob:
		image = image[:,:,::-1,:]
		zlocs = (im_z_size-1)-zlocs
		flag = True
	return image, zlocs, flag

def stack_augmentation(image, mask, zlocs, centroids, final_z_size):
	#VERY SLOW, need to eliminate the deep copies!
	image_og = copy.deepcopy(image)
	image_z_size = image_og.shape[2]
	mask_og = copy.deepcopy(mask)
	zlocs_og = copy.deepcopy(zlocs)
	centroids_og = copy.deepcopy(centroids)
	for f in range(int(np.floor(final_z_size/ image.shape[2]))-1):
		#flip with z
		im, znew, _ = image_zflip(image_og, image_z_size, zlocs_og, 0.5)
		znew += image_z_size
		#rotate?
		flag = True
		counter=0
		while flag:
			if counter==0:
				counter+=1
				im, m, centnew, flag = image_mask_rotate(im, mask_og, centroids_og, 0.33)
			else:
				im, m, centnew, flag = image_mask_rotate(im, m, centnew, 0.33)

		im, m, centnew, _ = image_xyflip(im, m, centnew, 0.5)

		image, mask, _ = stack_image(image, im, mask, m, z_to = final_z_size)
		#append centroids
		centroids = np.concatenate([centroids,centnew])
		#append z-locs
		zlocs = np.concatenate([zlocs, znew])

	return image, mask, zlocs, centroids

def stack_image(image, image_stack, mask, mask_stack, z_to, rotate = False):
	"""
	INPUTS
	-----------------------------------------------------
	image, image_stack = numpy array from dataset.load_image with shape H x W x Z x 1

	z_to = integer : from config.INPUT_Z, specifies how large to pad the output
			image in the axial dimension.

	mask, mask_stack = A boolean array of shape [height, width, N]

	rotate = boolean : specify True if you wish to add rotation before stacking.

	"""
	n_rot = 0
	if image.shape[2] + image_stack.shape[2] <= z_to:
		if rotate:
			#rotate image_stack 90 deg
			image_stack = np.rot90(image_stack)
			mask_stack = np.rot90(mask_stack)
			while np.random.rand()<=0.33: #continue to rotate.
				n_rot += 1
				image_stack = np.rot90(image_stack)
				mask_stack = np.rot90(mask_stack)
		return np.concatenate([image,image_stack],axis=2), np.concatenate([mask,mask_stack],axis=2), n_rot
	else:
		return image, mask, n_rot

def dense_translate_augmentation(image, mask, zlocs, centroids, final_z_size, N=1, min_prob=0.5):
	"""
	Do the same operations as dense_augmentation, but we'll also add a translation.

	we would like to do this operation not just on one stack, but repeat the operation
	on several stacks.
	In fact, we should repeat the process as many as config.OUT_CHANNELS to make sure
	that it could properly separate cells into at least that many overlapping color channels.
	"""

	"""
	Change the first image, call it IM.
	For every subsequent n in range(N), add it to IM.
	"""
	image_og = copy.deepcopy(image)
	im_z_size = image_og.shape[2]
	pad_z_amount = final_z_size - im_z_size
	mask_og = copy.deepcopy(mask)
	zlocs_og = copy.deepcopy(zlocs)
	centroids_og = copy.deepcopy(centroids)
	###First image###
	image, zlocs, zflag1 = image_zflip(image, im_z_size, zlocs, min_prob)
	#rotate!
	image, mask, centroids, rotflag1 = image_mask_rotate(image, mask, centroids, min_prob)
	#flip!
	image, mask, centroids, xyflag1 = image_xyflip(image, mask, centroids, min_prob)
	###Secondary images###
	for n in range(N-1):
		im, znew, zflag2 = image_zflip(image_og, im_z_size, zlocs_og, min_prob)
		#pick any number between pad_z_amount and 0.
		pad_z = int(np.round(np.random.rand()*pad_z_amount))
		znew += pad_z
		#rotate!
		im, m, centnew, rotflag2 = image_mask_rotate(im, mask_og, centroids_og, min_prob)
		#flip!
		im, m, centnew, xyflag2 = image_xyflip(im, m, centnew, min_prob)
		if np.logical_or(zflag1!=zflag1,  np.any([rotflag1, xyflag1, rotflag2, xyflag2])):
			if image.shape[2]<im.shape[2]+pad_z:
				image_pad = int(np.abs(image.shape[2] - (im.shape[2]+pad_z)))
				image = np.pad(image, ((0,0),(0,0),(0,image_pad),(0,0)), 'constant')
			im = np.pad(im, ((0,0),(0,0),(pad_z,0),(0,0)), 'constant')
			if im.shape[2]<image.shape[2]:
				#pad im to image shape.
				im_pad = image.shape[2]-im.shape[2]
				im = np.pad(im, ((0,0),(0,0),(im_pad,0),(0,0)), 'constant')
			#put images together
			image = np.where(im>image, im, image)
			mask = np.concatenate([mask, m], axis=2)
			#append centroids
			centroids = np.concatenate([centroids,centnew])
			#append z-locs
			zlocs = np.concatenate([zlocs, znew])
	return image, mask, zlocs, centroids
