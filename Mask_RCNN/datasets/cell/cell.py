"""
Mask R-CNN
Train on the nuclei segmentation dataset from the
Kaggle 2018 Data Science Bowl
https://www.kaggle.com/c/data-science-bowl-2018/

Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

Edited for Cell Segmentation Dataset by Christopher Z. Eddy
contact: eddych@oregonstate.edu

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
	   the command line as such:

	# Train a new model starting from ImageNet weights
	python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=imagenet

	# Train a new model starting from specific weights file
	python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=/path/to/weights.h5

	# Resume training a model that you had trained earlier
	python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=last

	# Generate submission file
	python3 nucleus.py detect --dataset=/path/to/dataset --subset=train --weights=<last or /path/to/weights.h5>
"""

# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
# if __name__ == '__main__':
#     import matplotlib
#     # Agg backend runs without a display
#     matplotlib.use('Agg')
#     import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import tifffile

import os
import sys
import json
import datetime
import numpy as np
import skimage.io
import skimage.draw
from imgaug import augmenters as iaa
import scipy.stats
import tensorflow as tf
import faulthandler; faulthandler.enable()
import tensorflow.keras.layers as KL

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
from mrcnn import metrics
#from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
#if not os.path.isdir(DEFAULT_LOGS_DIR):
#    raise ImportError("'DEFAULT_LOGS_DIR' does not point to an exisiting directory.")
# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
#if not os.path.isdir(RESULTS_DIR):
#    raise ImportError("'RESULTS_DIR' does not point to an exisiting directory.")

# The dataset doesn't have a standard train/val split, so I picked
# a variety of images to serve as a validation set.
VAL_IMAGE_IDS = []

############################################################
#  Configurations
############################################################

class CellConfig(Config):
	"""Configuration for training on the nucleus segmentation dataset."""
	# Give the configuration a recognizable name
	NAME = "cell"

	#Apply dimensionality.
	INPUT_DIM = "3D"
	assert INPUT_DIM in ["2D", "3D"]
	#2D works, CE 11/27/20
	#2D works for inference as well, but 3D segfaults. Can we find out where.
	INPUT_Z = 27 #specify image padding size.

	OUT_CHANNELS = 3 #this flag is only used in the dense/stack augmentations.

	#set the following flag for True if you wish to train the stem branch of the network first.
	#this only applies if the input_dim is 3D
	train_stem=True

	# Adjust depending on your GPU memory
	IMAGES_PER_GPU = 2 #this varies the batch size. Hopefully. IMAGES_PER_GPU

	# Number of classes (including background)
	NUM_CLASSES = 1 + 1  # Background + nucleus

	# Number of training and validation steps per epoch
	#I'm guessing 657 are the number of images they had.
	STEPS_PER_EPOCH = min(100, 500 // IMAGES_PER_GPU) #(657 - len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU
	VALIDATION_STEPS = max(1, 100 // IMAGES_PER_GPU)

	# Don't exclude based on confidence. Since we have two classes
	# then 0.5 is the minimum anyway as it picks between nucleus and BG
	DETECTION_MIN_CONFIDENCE = 0

	# Backbone network architecture
	# Supported values are: resnet50, resnet101
	BACKBONE = "resnet50"

	# Input image resizing
	# Random crops of size 512x512
	IMAGE_RESIZE_MODE = "centroid"
	IMAGE_MIN_DIM = 512
	IMAGE_MAX_DIM = 512
	# Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
	# up scaling. For example, if set to 2 then images are scaled up to double
	# the width and height, or more, even if MIN_IMAGE_DIM doesn't require it.
	# However, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
	#default was 0.
	IMAGE_MIN_SCALE = 1.0
	#with a setting of 512 for dims, and scale of 2, that means we will look at
	#1/4 of the image at any point.

	# Length of square anchor side in pixels. 8 is much smaller than we need.
	RPN_ANCHOR_SCALES = (8, 32, 64, 128, 256)

	# ROIs kept after non-maximum supression (training and inference)
	POST_NMS_ROIS_TRAINING = 1000
	POST_NMS_ROIS_INFERENCE = 2000

	# Non-max suppression threshold to filter RPN proposals.
	# You can increase this during training to generate more propsals.
	RPN_NMS_THRESHOLD = 0.9

	# How many anchors per image to use for RPN training
	RPN_TRAIN_ANCHORS_PER_IMAGE = 128

	# Image mean (RGB)
	if INPUT_DIM=="2D":
		MEAN_PIXEL = np.array([0.0,0.0,0.0]) #background was already taken off.
	else:
		MEAN_PIXEL = np.array([0.0,0.0,0.0,0.0])

	#np.array([43.53, 39.56, 48.22])

	# If enabled, resizes instance masks to a smaller size to reduce
	# memory load. Recommended when using high-resolution images.
	USE_MINI_MASK = True
	MINI_MASK_SHAPE = (128,128)#(56, 56)  # (height, width) of the mini-mask

	# Number of ROIs per image to feed to classifier/mask heads
	# The Mask RCNN paper uses 512 but often the RPN doesn't generate
	# enough positive proposals to fill this and keep a positive:negative
	# ratio of 1:3. You can increase the number of proposals by adjusting
	# the RPN NMS threshold.
	TRAIN_ROIS_PER_IMAGE = 128

	# Maximum number of ground truth instances to use in one image
	MAX_GT_INSTANCES = 200

	# Max number of final detections per image
	DETECTION_MAX_INSTANCES = 400

    #Projection Enhancement Network (PEN) options! "collect" is how the wide inception inspired
	#convolution branches are put back together to a 3-channel output image for entry into
	#CellPose network (options are "conv", "mean", or "max"). Kernels represent the varying
	#width of the PEN network; each kernel is a size for a convolution on the original image.
	#"block_pool" is how in each branch of the network the convolution should be pooled; options
	# are "conv" and "max". "block_filters" are the number of learned filters in each branch of
	# PEN in the first 3D convolution layer. For a 12gB GPU, with CellPose, block_filters can
	# only be as much as 3.
	PEN_opts = {
		"collect": "conv",
		"kernels": [1,3,5,7,11],
		"block_pool": "conv",
		"block_filters": 3
	}



class CellInferenceConfig(CellConfig):
	# Set batch size to 1 to run one image at a time
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	USE_MINI_MASK = False
	IMAGE_MAX_DIM = 1024 #image size.
	INPUT_Z = 27
	# Don't resize imager for inferencing
	#IMAGE_RESIZE_MODE = "pad64"
	# Non-max suppression threshold to filter RPN proposals.
	# You can increase this during training to generate more propsals.
	RPN_NMS_THRESHOLD = 0.7
	#2D works, CE 11/27/20
	#2D works for inference as well, but 3D segfaults. Can we find out where.
	IMAGE_RESIZE_MODE = "square"#"none"
	# Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
	# up scaling. For example, if set to 2 then images are scaled up to double
	# the width and height, or more, even if MIN_IMAGE_DIM doesn't require it.
	# However, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
	#default was 0.
	#IMAGE_MIN_SCALE = 1.0

class CellTestingConfig(CellInferenceConfig):
	Testing = True



############################################################
#  Dataset
############################################################

class CellDataset(utils.Dataset):

	def load_cell(self, dataset_dir, subset=None):
		"""Load a subset of the cell dataset.

		dataset_dir: Root directory of the dataset
		subset: Subset to load. Either the name of the sub-directory,
				such as stage1_train, stage1_test, ...etc. or, one of:
				* train: stage1_train excluding validation images
				* val: validation images from VAL_IMAGE_IDS
		dimensionality: string, either "2D" or "3D". Specifies how to
				load dataset and whether to include the stem branch.
		"""
		# Add classes. We have one class.
		# Naming the dataset nucleus, and the class nucleus
		self.add_class("cell", 1, "cell")

		# Which subset?
		# "val": use hard-coded list above
		# "train": use data from stage1_train minus the hard-coded list above
		# else: use the data from the specified sub-directory
		#assert subset in ["train", "val", "stage1_train", "stage1_test", "stage2_test"]
		if subset:
			#subset_dir = subset#"stage1_train" if subset in ["train", "val"] else subset
			load_dir = os.path.join(dataset_dir, subset, "images")
		else:
			load_dir = os.path.join(dataset_dir, "images")
		#if subset == "val":
		#    image_ids = VAL_IMAGE_IDS
		#else:
		# Get image ids from directory names
		image_ids = os.listdir(load_dir)
		image_ids = [x for x in image_ids if x[0]!="."]
		#next(os.walk(load_dir))[2] #this returns directory names inside dataset_dir.
		#sort image_ids
		image_ids.sort()
		#we should further filter these if they are "." files.
		#image_ids = [x for i,x in enumerate(image_ids) if x[0]!="."]
		#they should be called fname.ome.tif files.
		#returns "gt" and "images"
		#if subset == "train":
		#    image_ids = list(set(image_ids))#list(set(image_ids) - set(VAL_IMAGE_IDS))

		# Add images
		for image_id in image_ids:
			if subset:
				self.add_image(
					"cell",
					image_id=image_id[0:image_id.find(".")],
					path=os.path.join(dataset_dir,subset))#,"images",image_id))#os.path.join(dataset_dir, image_id, "images/{}.png".format(image_id)))
			else:
				self.add_image(
					"cell",
					image_id=image_id[0:image_id.find(".")],
					path=os.path.join(dataset_dir))

	def load_mask(self, image_id):
		"""Generate instance masks for an image.
	   Returns:
		masks: A bool array of shape [height, width, instance count] with
			one mask per instance.
		class_ids: a 1D array of class IDs of the instance masks.
		"""
		# If not a balloon dataset image, delegate to parent class.
		image_info = self.image_info[image_id]
		if image_info["source"] != "cell":
			return super(self.__class__, self).load_mask(image_id)
			#see config.py for parent class default load_mask function

		# Get mask directory from image path
		mask_dir = os.path.join(image_info['path'], "gt")
		#os.path.join(os.path.dirname(os.path.dirname(image_info['path'])), "gt")

		data = load_json_data(os.path.join(mask_dir,image_info['id']+".json")) #load file with same name.
		# Convert polygons to a bitmap mask of shape
		# [height, width, instance_count]
		mask = np.zeros([data["images"]["height"], data["images"]["width"], len(data['annotations']['regions']['area'])],
						dtype=np.uint8)
						#puts each mask into a different channel.
		for i,[verty,vertx] in enumerate(zip(data['annotations']['regions']['x_vert'],data['annotations']['regions']['y_vert'])):
			#alright, so this appears backwards (vertx, verty) but it is this way because of how matplotlib does plotting.
			#I have verified this notation is correct CE 11/20/20
			poly = np.transpose(np.array((vertx,verty)))
			rr, cc = skimage.draw.polygon(poly[:,0], poly[:,1], mask.shape[0:-1])
			try:
				mask[rr,cc,i] = 1
			except:
				print("too many objects, needs debugging")
				print(self.image_info[image_id])
			#put each annotation in a different channel.

		# Return mask, and array of class IDs of each instance. Since we have
		# one class ID only, we return an array of 1s
		return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

	def load_weight_map(self, image_id):
		"""Load unannotated regions so they do not contribute to loss
		"""
		# If not a balloon dataset image, delegate to parent class.
		image_info = self.image_info[image_id]
		if image_info["source"] != "cell":
			return super(self.__class__, self).load_mask(image_id)
			#see config.py for parent class default load_mask function

		# Get mask directory from image path
		try:
			mask_dir = os.path.join(image_info['path'], "gt")
			#os.path.join(os.path.dirname(os.path.dirname(image_info['path'])), "gt")

			data = load_json_data(os.path.join(mask_dir,image_info['id']+".json")) #load file with same name.
			# Convert polygons to a bitmap mask of shape
			# [height, width, instance_count]
			wmap = np.zeros([data["images"]["height"], data["images"]["width"],1],
							dtype=np.uint8)
							#puts each mask into a different channel.
			for verty,vertx in zip(data['pixelweight']['x_vert'],data['pixelweight']['y_vert']):
				#alright, so this appears backwards (vertx, verty) but it is this way because of how matplotlib does plotting.
				#I have verified this notation is correct CE 11/20/20
				poly = np.transpose(np.array((vertx,verty)))
				rr, cc = skimage.draw.polygon(poly[:,0], poly[:,1], wmap.shape[0:-1])
				wmap[rr,cc,0] = 1
				#put each annotation in a different channel.

			wmap = wmap.astype(np.bool)

		except:
			wmap = False #we dont' have shape yet. Still works with np.where.

		return wmap

	def load_detection(self, image_id):
		if not hasattr(self, 'detect_dir'):
			raise ValueError("'detect_dir' is not an attribute in dataset.")
		else:
			# if detection_dir is None:
			# 	if os.path.isdir(os.path.join(self.image_info[image_id]['path'],'detections')):
			# 		self.detect_dir = os.path.join(self.image_info[0]['path'],'detections')
			# 	else:
			# 		raise ValueError("'detections' directory does not exit.")
			# else:
			# 	if os.path.isdir(detection_dir):
			# 		self.detect_dir = detection_dir
			# 	else:
			# 		if os.path.isdir(os.path.join(self.image_info[0]['path'],detection_dir)):
			# 			self.detect_dir = os.path.join(self.image_info[0]['path'],detection_dir)
			# 		else:
			# 			raise ValueError("'{}' directory does not exit.".format(detection_dir))
			loaded = np.load(os.path.join(self.detect_dir,self.image_info[image_id]['id']+'.npz'))
			#keys: rois, class_ids, scores, masks
			#masks is H x W x N
			#rois is N x 4
			#class_ids is N vector
			#scores is N vector
			return loaded['masks']


	def load_image(self, image_id, dimensionality, mask=None, z_to=None):
		assert dimensionality in ["2D", "3D"], "'dimensionality' parameter not passed or not '2D' or '3D' in CellConfig"
		if dimensionality=="2D":
			"""Load the specified image and return a [H,W,3] Numpy array.
			This is one option. Alternatively, send in an [H,W,Z] numpy array,
			and do a simple 2D convolution in stem and learn 3 filters.
			"""
			# Load image
			#print(os.path.join(self.image_info[image_id]['path'],'images',self.image_info[image_id]['id']+'.ome.tif'))
			try:
				image = skimage.io.imread(os.path.join(self.image_info[image_id]['path'],'images',self.image_info[image_id]['id']+'.ome.tif'))
				#3d image. do a maximum projection.
				image = np.max(image,axis=0)
			except:
				image = skimage.io.imread(os.path.join(self.image_info[image_id]['path'],'images',self.image_info[image_id]['id']+'.tif'))

			# If has an alpha channel, remove it for consistency
			# if image.shape[-1] == 4:
			# 	image = image[..., :3]

			image = image.astype(np.float32)

			if np.max(image)<=1.:
				image = image*255.

			mean_val = scipy.stats.tmean(image.ravel(),(0,100))
			image = image - mean_val
			image[image<0]=0
			# If grayscale. Convert to RGB for consistency.
			if image.ndim != 3:
				image = skimage.color.gray2rgb(image)

			if mask is not None:
				bad_pixels = self.load_weight_map(image_id)
				#now
				#import pdb; pdb.set_trace()
				mask = np.max(mask,axis=-1) #take max projection
				mask = np.expand_dims(mask,axis=-1) #add dimension for np.where
				bad_pixels=np.where(mask==True,False,bad_pixels) #anywhere an annotated object is, we don't want to cancel it ou
				#for each channel in image, set these to the mode of image.
				#determine the mean of small numbers.
				image = np.where(bad_pixels==True, 0.0, image)
			#image output shape is [H,W,3]
			image = image / 255.#np.max(image)
		else:
			"""Load the specified image and return a [H,W,Z,1] Numpy array.
			"""
			assert z_to, "'z_to' parameter not passed from Config"
			#print(os.path.join(self.image_info[image_id]['path'],'images',self.image_info[image_id]['id']+'.ome.tif'))
			#we'll fix z
			#z_to = 15
			#ultimately, we'll do enough convolutions to get that down to the correct size.
			image = skimage.io.imread(os.path.join(self.image_info[image_id]['path'],'images',self.image_info[image_id]['id']+'.ome.tif'))

			#convert to float.
			image = image.astype(np.float32)

			if np.max(image)<=1.:
				image = image*255.0

			#####This will center pad the image, always#####
			#should be a 3D image with the z stack in the first channel.
			z_chan = image.shape[0]
			z_fill = z_to - z_chan
			#split in 2, round.
			z_before = z_fill//2
			z_after = z_fill - z_before
			##making new aray and filling it is faster than using pad, but only if we use "zeros" and not "full".
			##for a nonzero padding value, it is slower this way.
			image = image.astype(np.float32)
			pad_val = scipy.stats.tmean(image.ravel(),(0,100)) #notice we are excluding the cell objects.
			image = image - pad_val
			image[image<0]=0 #clip values. #this clip values was at 1 before.
			######CHANGED CE 05/15/22##########
			#img_temp = np.zeros(shape=(z_to,image.shape[1],image.shape[2]), dtype=image.dtype)
			#img_temp[z_before:z_before+z_chan,:,:] = image
			#image = img_temp
			###################################

			##Padding current array
			#pad_val = scipy.stats.tmean(image.ravel(),(0,100))
			##find the pad val, subtract the mean.
			#image = np.pad(image, ((z_before,z_after),(0,0),(0,0)), 'constant', constant_values=pad_val)
			#roll axis.
			image = np.rollaxis(image, 0, 3)

			if mask is not None:
				#load weight map
				bad_pixels = self.load_weight_map(image_id)
				#import pdb; pdb.set_trace()
				mask = np.max(mask,axis=-1) #take max projection
				mask = np.expand_dims(mask,axis=-1) #add dimension for np.where
				bad_pixels=np.where(mask==True,False,bad_pixels) #anywhere an annotated object is, we don't want to cancel it out.
				#for each channel in image, set these to the mode of image.
				#determine the mean of small numbers.
				image = np.where(bad_pixels==True, 0.0, image)

			image = np.expand_dims(image, axis=-1) #add so Channel is "gray"
			#image output is shape=[H,W,Z,1]
			#the default for conv layers is channels last. i.e. input is [Batch_size, H, W, Z, CH]
			image = image / 255.0#np.max(image)

		return image

	def load_z_positions(self, image_id):
		# If not a balloon dataset image, delegate to parent class.
		image_info = self.image_info[image_id]
		# Get mask directory from image path
		json_dir = os.path.join(image_info['path'], "gt")
		#os.path.join(os.path.dirname(os.path.dirname(image_info['path'])), "gt")
		data = load_json_data(os.path.join(json_dir,image_info['id']+".json"))
		image_z_size = data['images']['slices']
		zlocs = np.array(data['annotations']['regions']['approx_z'])
		#use x-y-z location to do PCA to transform data.
		cents = np.array(data['annotations']['regions']['centroid'])
		imsize = [data['images']['height'], data['images']['width'], image_z_size]
		return cents, zlocs, imsize

	def pad_z_image(self, image, z_to, z_begin=None, center=False, random_pad=True):
		"""
		INPUTS
		-----------------------------------------------------
		image = numpy array from dataset.load_image with shape H x W x Z x 1

		z_to = integer : from config.INPUT_Z, specifies how large to pad the output
				image in the axial dimension.

		center = boolean : specifies if you wish image to be edge padded with zeros.

		random_pad = boolean : specifies if you wish padding to be randomly applied to edges.

		z_begin = integer : specifies the number of padded images to add at the beginning
				of the image stack; the output size determined by config.INPUT_Z dictates
				the remaining padded images to be added at the end of the image stack.

		OUTPUTS
		-----------------------------------------------------
		image = numpy array with edge padding with zeros up to z_to according to padding
				strategy determined by input arguments.
		"""
		#image should be shape Z x H x W
		if image.shape[2] > image.shape[0]:
			print("The shape of input is Z x H x W x 1 rather than H x W x Z x 1")
			import pdb;pdb.set_trace()
		"""
		CE: 01/30/22 Changing to random_pad made it so the network did not learn
		the z-separation in the color channels out of PEN when analyzing spheroids
		(see cellpose20220129T1545). However, the loss weights could have been the problem (4, 1, 1, 1).
		"""
		if random_pad and z_begin is None:
			#should be a 3D image with the z stack in the first channel.
			z_chan = image.shape[2]
			z_fill = z_to - z_chan
			z_before = np.random.randint(0,z_fill)
		if center and z_begin is None:
			#should be a 3D image with the z stack in the first channel.
			z_chan = image.shape[2]
			z_fill = z_to - z_chan
			#split in 2, round.
			z_before = z_fill//2
			#z_after = z_fill - z_before
		if z_begin is not None:
			#should be a 3D image with the z stack in the first channel.
			z_chan = image.shape[2]
			if z_begin > z_to - z_chan:
				z_begin = z_to-z_chan #place images at the end.
			z_before = z_begin

		img_temp = np.zeros(shape=(image.shape[0],image.shape[1],z_to,1), dtype=image.dtype)
		img_temp[:,:,z_before:z_before+z_chan,:] = image
		image = img_temp
		return image

	def image_reference(self, image_id):
		"""Return the path of the image."""
		info = self.image_info[image_id]
		if info["source"] == "cell":
			return info["id"]
		else:
			super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################

def load_json_data(pth):
	with open(pth) as f:
		data=json.load(f)
	return data

def train(model, dataset_dir):
	"""Train the model."""
	# Training dataset.
	print("Loading the train set")
	dataset_train = CellDataset()
	dataset_train.load_cell(dataset_dir, "train")#, model.config.INPUT_DIM)
	dataset_train.prepare()

	# Validation dataset
	print("Loading the validation set")
	dataset_val = CellDataset()
	dataset_val.load_cell(dataset_dir, "val")
	dataset_val.prepare()

	# Image augmentation
	# http://imgaug.readthedocs.io/en/latest/source/augmenters.html

	augmentation = True

	#okay, so when making the model we have passed CellConfig (which becomes config)
	#which contains config.INPUT_DIM which then is passed on in model.train (passes self - which contains contains config)
	#which loads data_generator (passes config as an argument)
	#passed in load_image_gt, which passes config as an argument onto load_image
	#which takes dimensionality=config.INPUT_DIM as an argument
	#so in short, the type of data is passed accordingly


	# print("Training Mask-RCNN layers for 15 epochs with fixed PEN...")
	# model.train(dataset_train, dataset_val,
	# 			learning_rate=config.LEARNING_RATE,
	# 			epochs=15,
	# 			augmentation=augmentation,
	# 			layers='mask_rcnn')

	print("TRAINING ALL LAYERS for 50 epochs...")
	model.train(dataset_train, dataset_val,
				learning_rate=config.LEARNING_RATE,
				epochs=50,
				augmentation=augmentation,
				layers='all')

	#
	# if model.config.INPUT_DIM=="3D":
	#     if model.config.train_stem:
	#         print("Training stem branch for 10 epochs")
	#         model.train(dataset_train, dataset_val,
	#                     learning_rate=config.LEARNING_RATE,
	#                     epochs=20,
	#                     augmentation=augmentation,
	#                     layers='stem')
	#
	#     print("TRAINING ALL LAYERS for 25 epochs")
	#     model.train(dataset_train, dataset_val,
	#                 learning_rate=config.LEARNING_RATE,
	#                 epochs=25,
	#                 augmentation=augmentation,
	#                 layers='all')
	# else:
	#     print("TRAINING ALL LAYERS for 25 epochs")
	#     model.train(dataset_train, dataset_val,
	#                 learning_rate=config.LEARNING_RATE,
	#                 epochs=25,
	#                 augmentation=augmentation,
	#                 layers='all')
	#Note that the epochs build. That is, the first training sets model.epoch to 10.
	#then the next training in stem will train to model.epoch=20. Then, the final training goes for 10 more epochs



############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
	"""Encodes a mask in Run Length Encoding (RLE).
	Returns a string of space-separated values.
	"""
	assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
	# Flatten it column wise
	m = mask.T.flatten()
	# Compute gradient. Equals 1 or -1 at transition points
	g = np.diff(np.concatenate([[0], m, [0]]), n=1)
	# 1-based indicies of transition points (where gradient != 0)
	rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
	# Convert second index in each pair to lenth
	rle[:, 1] = rle[:, 1] - rle[:, 0]
	return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
	"""Decodes an RLE encoded list of space separated
	numbers and returns a binary mask."""
	rle = list(map(int, rle.split()))
	rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
	rle[:, 1] += rle[:, 0]
	rle -= 1
	mask = np.zeros([shape[0] * shape[1]], np.bool)
	for s, e in rle:
		assert 0 <= s < mask.shape[0]
		assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
		mask[s:e] = 1
	# Reshape and transpose
	mask = mask.reshape([shape[1], shape[0]]).T
	return mask


def mask_to_rle(image_id, mask, scores):
	"Encodes instance masks to submission format."
	assert mask.ndim == 3, "Mask must be [H, W, count]"
	# If mask is empty, return line with image ID only
	if mask.shape[-1] == 0:
		return "{},".format(image_id)
	# Remove mask overlaps
	# Multiply each instance mask by its score order
	# then take the maximum across the last dimension
	order = np.argsort(scores)[::-1] + 1  # 1-based descending
	mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
	# Loop over instance masks
	lines = []
	for o in order:
		m = np.where(mask == o, 1, 0)
		# Skip if empty
		if m.sum() == 0.0:
			continue
		rle = rle_encode(m)
		lines.append("{}, {}".format(image_id, rle))
	return "\n".join(lines)


############################################################
#  Detection
############################################################

# def detect(model, dataset_dir):
#     """Run detection on images in the given directory."""
#     print("Running on {}".format(dataset_dir))
#
#     # Create directory
#     if not os.path.exists(RESULTS_DIR):
#         os.makedirs(RESULTS_DIR)
#     submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
#     submit_dir = os.path.join(RESULTS_DIR, submit_dir)
#     os.makedirs(submit_dir)
#
#     # Read dataset
#     dataset = CellDataset()
#     dataset.load_cell(dataset_dir)
#     dataset.prepare()
#     #config = CellInferenceConfig()
#     # Load over images
#     submission = []
#     for image_id in dataset.image_ids:
#         print("reading image = ", image_id)
#         # Load image and run detection
#         #model is passed which is a class, with config as an object
#         image = dataset.load_image(image_id, dimensionality=model.config.INPUT_DIM)
#         #this likely has size 1024x1024xzx1 or 1024x1024x3, depending on config. Either way, we need to make it so the batch size is right.
#         # Detect objects
#         print('before model.detect = ',image.shape)
#         [r,molded_ims] = model.detect([image], verbose=0)
#         r=r[0]
#         print(molded_ims.shape)
#         m = molded_ims[0]
#         print('5')
#         if model.config.INPUT_DIM=="3D":
#             #take max projection with
#             x = visualize.show_max_proj(m,transform=False)
#             x = np.squeeze(x)
#             #should have shape 1024x1024x3
#             #renormalize to 255.
#             #x = x*255# puts it in same range as x.
#             #no, 0-1 is correct.
#             print(x.shape)
#         else:
#             x = m#image
#         # Encode image to RLE. Returns a string of multiple lines
#         print('6')
#         source_id = dataset.image_info[image_id]["id"]
#         rle = mask_to_rle(source_id, r["masks"], r["scores"])
#         submission.append(rle)
#         # Save image with masks
#         fig,ax = plt.subplots(1)
#         print('7')
#         visualize.display_instances(
#             x, r['rois'], r['masks'], r['class_ids'],
#             dataset.class_names, r['scores'],
#             show_bbox=True, show_mask=True, ax=ax,
#             title="Predictions")
#         plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]),dpi=1200)
#
#     # Save to csv file
#     submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
#     file_path = os.path.join(submit_dir, "submit.csv")
#     with open(file_path, "w") as f:
#         f.write(submission)
#     print("Saved to ", submit_dir)

def testing(m, dataset, args):
	if not m.config.Testing:
		print("Currently the testing configuration is not on. Remaking model with testing configuration from CellTestingConfig. Model is set to 'inference' mode.")
		config = CellTestingConfig()
		model = modellib.MaskRCNN(mode="inference", config=config,
								  model_dir=args.logs)
		load_weights(model,args.weights)

	else:
		model = m

	image_id = np.random.choice(dataset.image_ids)
	image = dataset.load_image(image_id,dimensionality=model.config.INPUT_DIM, z_to=model.config.INPUT_Z)
	images=[image]

	molded_images, image_metas, windows = model.mold_inputs(images)
	print(molded_images.shape)
	#we need image metas to look correct.
	#also, below in image_shape, we likely need that to be the (*molded_images[0][:2],3)? Should be 3 channel...

	# Validate image sizes
	# All images in a batch MUST be of the same size
	print('2')
	image_shape = molded_images[0].shape
	for g in molded_images[1:]:
		assert g.shape == image_shape,\
			"After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

	# Anchors
	anchors = model.get_anchors(np.array(list(image_shape[:2]+(model.config.IMAGE_CHANNEL_COUNT,))))
	#image_shape)
	# Duplicate across the batch dimension because Keras requires it
	# TODO: can this be optimized to avoid duplicating the anchors?
	anchors = np.broadcast_to(anchors, (model.config.BATCH_SIZE,) + anchors.shape)

	print(anchors.shape)

	a = tf.constant([[1, 2], [3, 4]])
	if model.config.INPUT_DIM=="3D":
		detections_call, _, _, mrcnn_mask_call, _, _, _, stem_out_call, input_image_call =\
			model.keras_model([molded_images, image_metas, anchors], training=False)#, verbose=0)
		in_max = input_image_call.eval(session=tf.compat.v1.Session())
		#import pdb; pdb.set_trace()
		send_out = stem_out_call.eval(session=tf.compat.v1.Session())
		#in_max = in_max[0]
		#in_max = np.max(in_max,axis=2)
		#okay, so the images are good now.
	else:
		detections_call, _, _, mrcnn_mask_call, _, _, _, input_image_call=\
			model.keras_model([molded_images, image_metas, anchors], training=False)#, verbose=0)
		in_max = input_image_call.eval(session=tf.compat.v1.Session())
	#import pdb; pdb.set_trace()
	# detections, _, _, mrcnn_mask, _, _, _, stem_out =\
	#     self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
	# detections, _, _, mrcnn_mask, _, _, _=\
	#     self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
	#the above segfaults for 3D input
	# Process detections
	#out.eval(session=tf.compat.v1.Session())
	#a = a.eval(session=tf.compat.v1.Session()) #works.
	#the following error out.
	#stem_out = stem_out.eval(session=tf.compat.v1.Session())
	# detections = detections_call.eval(session=tf.compat.v1.Session())
	# mrcnn_mask = mrcnn_mask_call.eval(session=tf.compat.v1.Session())
	return in_max, image_metas

def load_weights(model,weights_path,exclude=None):
	# Select weights file to load
	import h5py
	if h5py is None:
		raise ImportError('"load_weights" requires h5py.')
	#if args.weights == "coco":
	if weights_path == COCO_WEIGHTS_PATH:
		# Download weights file
		if not os.path.exists(weights_path):
			utils.download_trained_weights(weights_path)
		# Exclude the last layers because they require a matching
		# number of classes
		exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"]
	# elif args.weights.lower() == "last":
	#     # Find last trained weights
	#     weights_path = model.find_last()
	# elif args.weights.lower() == "imagenet":
	#     # Start from ImageNet trained weights
	#     weights_path = model.get_imagenet_weights()
	# else:
	# 	weights_path = args.weights

	print("\nLoading weights from {}...".format(weights_path))
	print("\n")

	initial_weights = [layer.get_weights() for layer in model.layers]

	if exclude:
		#first, find the layers that have weights in the model.
		layers = [a.name for a in model.layers if a.get_weights()]
		#second, filter out any that are in exclude.
		layers = list(filter(lambda l: l not in exclude, layers))
		#now, we need to find the names of layers in the h5 file, and exclude any that also are not in layers.
		data=h5py.File(weights_path,mode="r")
		h5_layers = list(data.keys())
		h5_layers = [f for f in h5_layers if f in layers]
		#h5_layer_inds = [i for i,name in enumerate(list(data.keys())) if name in layers]
		#so, now the long way.
		#now, load the weights that are in layers.
		#pull the weights of those layers
		#weights_list = [a.get_weights() for a in model.layers if a.name in layers] #this is actually weights from model. Not what we want to load.
		#layer_inds =[i for i,a in enumerate(model.layers) if a.name in h5_layers]
		#find the corresponding layer index for the h5 layer.
		model_layers = [a.name for a in model.layers]
		model_layers_inds = [h if j<1 else "error" for h5_layer in h5_layers for j,h in enumerate([i for i,x in enumerate(model_layers) if x==h5_layer])]
		#there should be no errors.
		if any(isinstance(s, str) for s in model_layers_inds):
			raise Exception("Error in matching h5 file layer to model layer")

		for i,model_i in enumerate(model_layers_inds):
			att_items=[b for a,b in data[h5_layers[i]].attrs.items()][0]
			w=[]
			for sub_item in att_items:
				w.append(np.array(data[h5_layers[i]][sub_item.decode("utf-8")],dtype="float32"))
			#so now, what if w has different shapes than the current model layer?
			#should we pass, or throw a warning?
			#likely pass.
			#I'm not going to code this in yet. We'll let it error if there is a problem.
			try:
				model.layers[model_i].set_weights(w)
			except:
				print("Model layer and h5 layer have different shape")
				print("Model layer name = {}, h5 layer name = {}".format(model_layers[model_i],h5_layers[i]))
				print("len model layer = {}, len h5 layer = {}".format(len(model_layers[model_i]),len(h5_layers[i])))
				for item in model_layers[model_i]:
					print("submodule model layer size = ", item.shape)
				for item in h5_layers[i]:
					print("submodule h5 layer size = ", item.shape)
				import pdb; pdb.set_trace()
				raise Exception("Model and h5 layer have different weight shapes!")

	else:
		#model.keras_model.trainable=True #fix weights problem.
		model.load_weights(weights_path, by_name=True)
		#model.keras_model.trainable=False

	for layer,initial in zip(model.layers,initial_weights):
		#import pdb; pdb.set_trace()
		weights = layer.get_weights()
		if weights and all(tf.nest.map_structure(np.array_equal,weights,initial)):
			print(f'Checkpoint contained no weights for layer {layer.name}!')
	print("done loading weights")
	#do we need to return model now? Or will the weights just stick with the model. I think they should stick.
	#nope, they stick just fine :)

def testing_train_generator(model, train_dataset):
	#import pdb; pdb.set_trace()
	train_generator = modellib.data_generator(train_dataset, model.config, shuffle=True,\
									 augmentation=True,\
									 batch_size=model.config.BATCH_SIZE,\
									 no_augmentation_sources=None)
	[inputs, outputs] = next(train_generator)

	# inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
	#           batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
	# outputs = []

	return [inputs,outputs]

def test_augment(image,mask):
	import imgaug
	import imgaug.augmenters as iaa

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
	#roll axis
	image_out = np.rollaxis(image, 2, 0) #this may flip the image...
	#image_rot = np.rollaxis(image, 1, 3)
	#import pdb; pdb.set_trace()
	image_out = det.augment_images(image_out)
	mask_out = det.augment_image(mask.astype(np.uint8),
							 hooks=imgaug.HooksImages(activator=hook))
	#roll axis back
	image_out = np.rollaxis(image_out,0, 3)
	assert image_out.shape == image_shape, "Augmentation shouldn't change image size"
	assert mask_out.shape == mask_shape, "Augmentation shouldn't change mask size"
	# Change mask back to bool
	mask_out = mask_out.astype(np.bool)

	return [image_out, mask_out]

def Stem_test(image_shape, config, args):
	#create a mini model of just the stem. Then send in image to see the output and convert it to a numpy array.
	inp = tf.keras.Input(shape=image_shape)
	#x = modellib.stem_project_test(inp)
	#stem_project_test(inp)#stem_graph_max(inp) #this works. Use weights cell20201214T1328
	x = modellib.stem_graph_max(inp, batch_size=1, z_slices=config.INPUT_Z, PEN_opts=config.PEN_opts, train_bn=config.PEN_opts) #does features only. See cell20201214T1214 #not working.
	#stem_graph_max_input(inp) #does features and max. See cell20201214T2044
	# x should now be a bn x 1024 x 1024 x 3 image
	model_test = tf.keras.models.Model([inp], x, name='stem_test')
	#load the weights of the model

	initial_weights = [layer.get_weights() for layer in model_test.layers]
	#model_test.load_weights(args.weights, by_name=True)
	load_weights(model_test,args.weights)
	for layer,initial in zip(model_test.layers,initial_weights):
		#import pdb; pdb.set_trace()
		weights = layer.get_weights()
		if weights and all(tf.nest.map_structure(np.array_equal,weights,initial)):
			print(f'Checkpoint contained no weights for layer {layer.name}!')
	print("done loading weights")

	#should be good to go:
	return model_test

def view_features_model(config, image_shape, args, with_stem=True):
	#create a mini model of just the stem. Then send in image to see the output and convert it to a numpy array.
	inp = tf.keras.Input(shape=image_shape)

	#x = modellib.stem_graph_max_input(inp)
	x = modellib.stem_graph_max(inp, batch_size=1, z_slices=config.INPUT_Z, PEN_opts=config.PEN_opts, train_bn=False)
	#stem_project_test(inp)#stem_graph_max(inp) #this works. Use weights cell20201214T1328
	#stem_graph_max(inp) #does features only. See cell20201214T1214 #not working.
	#stem_graph_max_input(inp) #does features and max. See cell20201214T2044
	# x should now be a bn x 1024 x 1024 x 3 image
	C1, C2, C3, C4, C5 = modellib.resnet_graph(x, config.BACKBONE,
									 stage5=True, train_bn=config.TRAIN_BN)
	resnet_outputs = [x, C1, C2, C3, C4, C5]
	inputs = [inp]

	model_test = tf.keras.models.Model(inputs, resnet_outputs, name='feature_model')
	#load the weights of the model

	initial_weights = [layer.get_weights() for layer in model_test.layers]
	#model.load_weights(weights_path, by_name=True)
	#model_test.load_weights(args.weights, by_name=True)
	load_weights(model_test,args.weights)
	for layer,initial in zip(model_test.layers,initial_weights):
		#import pdb; pdb.set_trace()
		weights = layer.get_weights()
		if weights and all(tf.nest.map_structure(np.array_equal,weights,initial)):
			print(f'Checkpoint contained no weights for layer {layer.name}!')
	print("done loading weights")

	return model_test


def view_features(config, image, args, with_stem=True):
	image_shape = image.shape
	image, window, scale, padding, crop = utils.resize_image(image, centroids=None, mode='none')
	model = view_features_model(config, image_shape, args, with_stem=with_stem)
	#above loads weights. Needs arguments passed to it.
	#expand dimensionality of image to include batch size.
	image = np.expand_dims(image,axis=0)
	[stem_out, c1,c2,c3,c4,c5] = model.predict([image],verbose=0)
	resnet_output = [stem_out,c1,c2,c3,c4,c5]
	return [resnet_output,model]

def view_stem_output(image, config, args):
	image_shape = image.shape
	image, window, scale, padding, crop = utils.resize_image(image, centroids=None, mode='none')
	model = Stem_test(image_shape, config, args)
	image = np.expand_dims(image,axis=0)
	[stem_out] = model.predict([image],verbose=0)
	return [stem_out,model]


def detect_with_stem_model(config,weights_path):
	#create a mini model of just the stem. Then send in image to see the output and convert it to a numpy array.
	#inp = tf.keras.Input(shape=(config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM, config.INPUT_Z, 1))
	inp = tf.keras.Input(shape=(None,)*2 + (config.INPUT_Z, 1))
	input_image_meta = tf.keras.Input(shape=[config.IMAGE_META_SIZE])

	#x = modellib.stem_graph_max_input(inp)
	#x = modellib.stem_project_test(inp)#stem_graph_max(inp) #this works. Use weights cell20201214T1328
	x = modellib.stem_graph_max(inp, batch_size=1, z_slices=config.INPUT_Z, PEN_opts=config.PEN_opts, train_bn=False) #does features only. See cell20201214T1214 #not working.
	#x = modellib.stem_graph_max_z(inp,config.INPUT_Z)
	#stem_graph_max_input(inp) #does features and max. See cell20201214T2044
	# x should now be a bn x 1024 x 1024 x 3 image
	C1, C2, C3, C4, C5 = modellib.resnet_graph(x, config.BACKBONE,
									 stage5=True, train_bn=config.TRAIN_BN)
	resnet_outputs = [C1, C2, C3, C4, C5]

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

	#outputs = [resnet_outputs, rpn_feature_maps]

	#anchors = input_anchors
	anchors = get_anchors(config,config.IMAGE_SHAPE)
	# Duplicate across the batch dimension because Keras requires it
	# TODO: can this be optimized to avoid duplicating the anchors?
	anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
	# A hack to get around Keras's bad support for constants
	#anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
	anchors = modellib.AnchorsLayer(anchors, name="anchors_inference")(x)
	rpn = modellib.build_rpn_model(config.RPN_ANCHOR_STRIDE,
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
	proposal_count = config.POST_NMS_ROIS_INFERENCE
	rpn_rois = modellib.ProposalLayer(
		proposal_count=proposal_count,
		nms_threshold=config.RPN_NMS_THRESHOLD,
		name="ROI",
		config=config)([rpn_class, rpn_bbox, anchors])

	mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
		modellib.fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
							 config.POOL_SIZE, config.NUM_CLASSES,
							 train_bn=config.TRAIN_BN,
							 fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

	# Detections
	# output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in
	# normalized coordinates
	detections = modellib.DetectionLayer(config, name="mrcnn_detection")(
		[rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

	detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)

	mrcnn_mask = modellib.build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
									  input_image_meta,
									  config.MASK_POOL_SIZE,
									  config.NUM_CLASSES,
									  train_bn=config.TRAIN_BN)

	outputs = [detections, mrcnn_mask]
	inputs = [inp, input_image_meta]

	model_test = tf.keras.models.Model(inputs, outputs, name='stem_test')
	#load the weights of the model

	initial_weights = [layer.get_weights() for layer in model_test.layers]
	#model_test.load_weights(weights_path, by_name=True)
	load_weights(model_test,weights_path)
	for layer,initial in zip(model_test.layers,initial_weights):
		#import pdb; pdb.set_trace()
		weights = layer.get_weights()
		if weights and all(tf.nest.map_structure(np.array_equal,weights,initial)):
			print(f'Checkpoint contained no weights for layer {layer.name}!')
	print("done loading weights")


	#should be good to go:
	return model_test


def detect_without_stem_model(config,weights_path):
	#create a mini model of just the stem. Then send in image to see the output and convert it to a numpy array.
	inp = tf.keras.Input(shape=(None,)*2 + (3,))
	#inp = tf.keras.Input(shape=(1024, 1024, 3))
	input_image_meta = tf.keras.Input(shape=[config.IMAGE_META_SIZE])
	#stem_project_test(inp)#stem_graph_max(inp) #this works. Use weights cell20201214T1328
	#stem_graph_max(inp) #does features only. See cell20201214T1214 #not working.
	#stem_graph_max_input(inp) #does features and max. See cell20201214T2044
	# x should now be a bn x 1024 x 1024 x 3 image
	C1, C2, C3, C4, C5 = modellib.resnet_graph(inp, config.BACKBONE,
									 stage5=True, train_bn=config.TRAIN_BN)
	resnet_outputs = [C1, C2, C3, C4, C5]

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

	#outputs = [resnet_outputs, rpn_feature_maps]

	#anchors = input_anchors
	anchors = get_anchors(config,config.IMAGE_SHAPE)
	# Duplicate across the batch dimension because Keras requires it
	# TODO: can this be optimized to avoid duplicating the anchors?
	anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
	# A hack to get around Keras's bad support for constants
	#anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
	anchors = modellib.AnchorsLayer(anchors, name="anchors_inference")(inp)
	rpn = modellib.build_rpn_model(config.RPN_ANCHOR_STRIDE,
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
	proposal_count = config.POST_NMS_ROIS_INFERENCE
	rpn_rois = modellib.ProposalLayer(
		proposal_count=proposal_count,
		nms_threshold=config.RPN_NMS_THRESHOLD,
		name="ROI",
		config=config)([rpn_class, rpn_bbox, anchors])

	mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
		modellib.fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
							 config.POOL_SIZE, config.NUM_CLASSES,
							 train_bn=config.TRAIN_BN,
							 fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

	# Detections
	# output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in
	# normalized coordinates
	detections = modellib.DetectionLayer(config, name="mrcnn_detection")(
		[rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

	detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)

	mrcnn_mask = modellib.build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
									  input_image_meta,
									  config.MASK_POOL_SIZE,
									  config.NUM_CLASSES,
									  train_bn=config.TRAIN_BN)

	outputs = [detections, mrcnn_mask]
	inputs = [inp, input_image_meta]

	model_test = tf.keras.models.Model(inputs, outputs, name='stem_test')
	#load the weights of the model

	initial_weights = [layer.get_weights() for layer in model_test.layers]
	#model_test.load_weights(weights_path, by_name=True)
	load_weights(model_test, weights_path)
	for layer,initial in zip(model_test.layers,initial_weights):
		#import pdb; pdb.set_trace()
		weights = layer.get_weights()
		if weights and all(tf.nest.map_structure(np.array_equal,weights,initial)):
			print(f'Checkpoint contained no weights for layer {layer.name}!')
	print("done loading weights")


	#should be good to go:
	return model_test

def detect(dataset, model, config, submit_dir, image_id=None):
	"""Run detection on images in the given directory."""
	#print("Running on {}".format(dataset_dir))
	#allresults=[]
	if image_id is None:
		for image_id,_ in enumerate(dataset.image_info):
			print("Running detection on image {} of {}".format(image_id+1, len(dataset.image_info)))
			if len(model.layers[0].output_shape[0][1:])>3:
				image = dataset.load_image(image_id=image_id,dimensionality="3D", z_to=config.INPUT_Z)
			else:
				image = dataset.load_image(image_id=image_id,dimensionality="2D")
			#previous loads the image for the given model, particular to the input shape.
			og_shape = image.shape[:2]

			#########################################################################
			######################### PAD THE IMAGE #################################
			#########################################################################
			#pad the images
			if config.INPUT_DIM=="3D":
				image = dataset.pad_z_image(image, z_to = config.INPUT_Z, center = True, random_pad = False)

			image, window, scale, padding, crop = utils.resize_image(image, centroids=None, mode=config.IMAGE_RESIZE_MODE, max_dim=config.IMAGE_MAX_DIM)

			active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
			source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
			active_class_ids[source_class_ids] = 1
			# Image meta data
			image_meta = modellib.compose_image_meta(1, (*image.shape[:2],3), (*image.shape[:2],3),
											window, scale, active_class_ids)

			#expand dimensionality of image to include batch size.
			image = np.expand_dims(image,axis=0)
			#print(image.shape)
			image_meta = np.expand_dims(image_meta,axis=0)
			#print(image_meta)
			#convert to tensor. It isn't clear to me where in model it does this for training...
			# it accepts numpy arrays, output is a numpy array.
			[detections, mrcnn_mask]=model.predict([image,image_meta],verbose=0)

			window = np.expand_dims(window,axis=0)
			results = return_detect_results(image, og_shape, detections, mrcnn_mask, window)
			results = results[0]
			#allresults.append(results)

			#results is a dictionary with keys "rois","class_ids","scores", and "masks".
			#masks are H x W x N, where N are the number of detections.
			"""
			Do we want to write this to a dictionary to an output file? Right now,
			we write an RGB output file. Instead, we could find the N annotations,
			we have their bounding boxes in rois, and just record the image size (HxW).
			Actually, it does look like we save the results files as an npz compressed
			file.
			"""

			#fig,ax1=plt.subplots(1)
			#display_results(results,image[0],ax=ax1)
			np.savez_compressed("{}/{}".format(submit_dir, dataset.image_info[image_id]["id"]), **results)
			#plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]),dpi=1200)
			#M = return_output_mask_rgb(image[0],results)
			#tifffile.imwrite("{}/{}.tif".format(submit_dir,dataset.image_info[image_id]["id"]),M)
	else:
		print("Running detection on image {}".format(dataset.image_info[image_id]["id"]))
		if len(model.layers[0].output_shape[0][1:])>3:
			image = dataset.load_image(image_id=image_id,dimensionality="3D",z_to=config.INPUT_Z)
		else:
			image = dataset.load_image(image_id=image_id,dimensionality="2D")

		og_shape = image.shape[:2]
		image, window, scale, padding, crop = utils.resize_image(image, centroids=None, mode=config.IMAGE_RESIZE_MODE, max_dim=config.IMAGE_MAX_DIM)

		active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
		source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
		active_class_ids[source_class_ids] = 1
		# Image meta data
		image_meta = modellib.compose_image_meta(1, (*image.shape[:2],3), (*image.shape[:2],3),
										window, scale, active_class_ids)

		#expand dimensionality of image to include batch size.
		image = np.expand_dims(image,axis=0)
		image_meta = np.expand_dims(image_meta,axis=0)
		#convert to tensor. It isn't clear to me where in model it does this for training...
		# it accepts numpy arrays, output is a numpy array.
		[detections, mrcnn_mask]=model.predict([image,image_meta],verbose=0)

		window = np.expand_dims(window,axis=0)
		results = return_detect_results(image, og_shape, detections, mrcnn_mask, window)
		results = results[0]

		fig,ax1=plt.subplots(1)
		display_results(results,image[0],ax=ax1)
		np.savez_compressed("{}/{}".format(submit_dir, dataset.image_info[image_id]["id"]), **results)
		plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]),dpi=1200)
		M = return_output_mask_rgb(image[0],results)
		tifffile.imwrite("{}/{}.tif".format(submit_dir,dataset.image_info[image_id]["id"]),M)

	#return results

def display_results(r,image,dataset=None,ax=None):
	if not ax:
		auto_show=True
		fig,ax = plt.subplots(1)
	else:
		auto_show=False
	if len(image.shape)>3:
		#take max projection.
		image = np.max(image,axis=2)
		#stack.
		image = np.concatenate((image,image,image),axis=2)
	if not dataset:
		visualize.display_instances(
			image, r['rois'], r['masks'], r['class_ids'],
			['BG','cell'], r['scores'],
			show_bbox=True, show_mask=True, ax=ax,
			title="Predictions")
	else:
		visualize.display_instances(
			image, r['rois'], r['masks'], r['class_ids'],
			dataset.class_names, r['scores'],
			show_bbox=True, show_mask=True, ax=ax,
			title="Predictions")
	if auto_show:
		plt.show()

def return_output_mask_rgb(image,r):
	"""
	Takes mask results (from results) with shape H x W x N
	and projects them onto a single image.
	"""
	masks=r['masks'] #shape H x W x N
	#project into zeros image and save in tifffile
	#import pdb;pdb.set_trace()
	all_masks = np.zeros(shape=image.shape[:2]+(3,),dtype=np.uint8)
	for i in range(masks.shape[2]):
		c=0
		added=0
		while added==0:
			assert c<3, "Too much overlap to return RGB mask image."
			if np.sum(np.ravel(all_masks[:,:,c]) * np.ravel(masks[:,:,i]))==0:
				added=1
				all_masks[:,:,c]=np.where(masks[:,:,i]==1, 255, all_masks[:,:,c])
			else:
				c=c+1
	return all_masks

def return_output_mask(image,r):
	"""
	Takes mask results (from results) with shape H x W x N
	and projects them onto a single image.
	"""
	masks=r['masks'] #shape H x W x N
	#project into zeros image and save in tifffile
	#import pdb;pdb.set_trace()
	all_masks = np.zeros(shape=image.shape[:2])
	for i in range(masks.shape[2]):
		all_masks=np.where(masks[:,:,i]==1, 1.0, all_masks)
	#import pdb;pdb.set_trace()
	return all_masks


#okay, so the filters look exactly the same at least for c1. They look the same for all these filters.
def compare_output_layers(L, f, N1, N2):
	#L = layer (1-5), f = filter number.
	#N1 = filters from mode 1, N2 = filters from mode 2
	if f>N1[L].shape[-1]:
		print("filter number (second arg) larger than size of layer filters")
	else:
		if N1[L].shape!=N2[L].shape:
			print("Different modes were selected")
		else:
			fig,(ax1,ax2)=plt.subplots(1,2)
			ax1.imshow(N1[L][0,:,:,f])
			ax2.imshow(N2[L][0,:,:,f])
			plt.show()

def see_output_filter(f, layer_output):
	fig,ax1 = plt.subplots(1)
	ax1.imshow(layer_output[0,:,:,f])
	plt.show()

def show_filters(layer_output):
	if np.shape(layer_output)[3]>=6:
		#goal, randomly select 6, non-zero filters form layer_output.
		fig,axs = plt.subplots(2,3)
		axs = axs.ravel()
		#find non-zero filters.
		#inds = [i for i in range(np.shape(layer_output)[3]) if sum(np.ravel(layer_output[0,:,:,i]))>0.0]
		#randomly select six of these indices.
		import random
		inds=[]
		while len(inds)<6:
			ind = random.choice(range(np.shape(layer_output)[3]))
			if sum(np.ravel(layer_output[0,:,:,ind])>0.0):
				inds.append(ind)
		#inds = [random.choice(range(np.shape(layer_output)[3])) if
		#inds = random.choices(inds, k=6)
		for i in range(6):
			axs[i].imshow(layer_output[0,:,:,inds[i]]/np.max(np.ravel(layer_output[0,:,:,inds[i]])))
			axs[i].set_axis_off()
			axs[i].set_title("filter = {}".format(inds[i]))
		plt.show()
	else:
		#goal, randomly select 6, non-zero filters form layer_output.
		fig,axs = plt.subplots(1,np.shape(layer_output)[3])
		axs = axs.ravel()
		#find non-zero filters.
		#inds = [i for i in range(np.shape(layer_output)[3]) if sum(np.ravel(layer_output[0,:,:,i]))>0.0]
		#randomly select six of these indices.
		import random
		max_val = np.max(np.ravel(layer_output[0,:,:,:]))
		for i in range(np.shape(layer_output)[3]):
			axs[i].imshow(layer_output[0,:,:,i]/max_val)
			axs[i].set_axis_off()
			axs[i].set_title("filter = {}".format(inds[i]))
		plt.show()

def return_detect_results(images, og_shape, detections,mrcnn_mask,windows):
	results = []
	for i, image in enumerate(images):
		final_rois, final_class_ids, final_scores, final_masks =\
			unmold_detections(detections[i], mrcnn_mask[i],
							  og_shape, image.shape,
							  windows[i])
		results.append({
			"rois": final_rois,
			"class_ids": final_class_ids,
			"scores": final_scores,
			"masks": final_masks,
		})
	return results

def compute_backbone_shapes(config, image_shape):
	"""Computes the width and height of each stage of the backbone network.
	Returns:
		[N, (height, width)]. Where N is the number of stages
	"""
	import math
	if callable(config.BACKBONE):
		return config.COMPUTE_BACKBONE_SHAPE(image_shape)

	# Currently supports ResNet only
	assert config.BACKBONE in ["resnet50", "resnet101"]
	return np.array(
		[[int(math.ceil(image_shape[0] / stride)),
			int(math.ceil(image_shape[1] / stride))]
			for stride in config.BACKBONE_STRIDES])

def get_anchors(config, image_shape):
	"""Returns anchor pyramid for the given image size."""
	backbone_shapes = compute_backbone_shapes(config, image_shape)

	a = utils.generate_pyramid_anchors(
		config.RPN_ANCHOR_SCALES,
		config.RPN_ANCHOR_RATIOS,
		backbone_shapes,
		config.BACKBONE_STRIDES,
		config.RPN_ANCHOR_STRIDE)
		# Keep a copy of the latest anchors in pixel coordinates because
		# it's used in inspect_model notebooks.
		# TODO: Remove this after the notebook are refactored to not use it
	anchors = utils.norm_boxes(a, image_shape[:2])
	return anchors

def unmold_detections(detections, mrcnn_mask, original_image_shape,
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
	#import pdb;pdb.set_trace()
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


###########################################################
""" CALCULATE METRICS FUNCTIONS"""
##########################################################
def measure_metrics(dataset, config, mask_thresh=0.5, size_thresh=50, alpha=0.5, detection_dir = None):
	"""
	fpath: string, file path to where training files are located.
		Structure:
		filepath /
		--> gt
			--> annotated .json files
		--> images
			--> .ome.tif files
		--> detections
			--> .npz files

		All should share the same filename structure before extensions!

	mask_thresh: float, threshold to apply on predicted masks

	boundary_thresh: float, threshold to apply on predicted boundaries

	size_thresh: float, threshold to apply on minimium object areas

	alpha: float or list of floats, minimum intersection over union value
		for which a possible true positive is decided.

	detection_dir: precise folder path or extension of self.dataset_dir where the
		detected .npz files are located.
	"""
	#check that the necessary files exist in filepath
	if not os.path.isdir(os.path.join(dataset.image_info[0]['path'], "gt")):
		print("mask directory does not exist at: {}".format(os.path.join(dataset.image_info[0]['path'], "gt")))
		raise ValueError("Mask directory does not exit.")
	###CHECK DETECTION DIRECTORY.
	if detection_dir is None:
		if os.path.isdir(os.path.join(dataset.image_info[0]['path'],'detections')):
			dataset.detect_dir = os.path.join(dataset.image_info[0]['path'],'detections')
		else:
			raise ValueError("'detections' directory does not exit.")
	else:
		if os.path.isdir(detection_dir):
			dataset.detect_dir = detection_dir
		else:
			if os.path.isdir(os.path.join(dataset.image_info[0]['path'],detection_dir)):
				dataset.detect_dir = os.path.join(dataset.image_info[0]['path'],detection_dir)
			else:
				raise ValueError("'{}' directory does not exit.".format(detection_dir))

	#if a single alpha value is given, turn to list.
	if isinstance(alpha,float):
		alpha = [alpha]

	all_ious = []
	all_intersections = []
	all_unions = []
	all_gt_mask_pix = []
	all_pred_mask_pix = []

	print("Importing images and calculating IoUs...")
	for image_id in range(len(dataset.image_info)):
		print("Importing {}...{}/{}".format(dataset.image_info[image_id]['id'], image_id+1, len(dataset.image_info)))
		gt_masks, pred_masks = modellib.load_image_gt_metrics(dataset, config, image_id)
		#now use metrics.functions to run metrics similar to cellpose!
		gt_mask_pix, pred_mask_pix = metrics.get_image_regions(pred_masks, gt_masks, size_thresh=size_thresh)
		#calculate the ious of the different regions, return iou array.
		iou_matrix, intersections, unions = metrics.calculate_IoUs_v2(gt_mask_pix, pred_mask_pix)
		#store the iou matrix
		all_ious.append(iou_matrix)
		all_intersections.append(intersections)
		all_unions.append(unions)
		all_gt_mask_pix.append(gt_mask_pix)
		all_pred_mask_pix.append(pred_mask_pix)


	all_matches = []
	all_image_AP = []
	zero_matches = metrics.pred_to_gt_assignment(all_ious, alpha=0.)
	for i,min_iou in enumerate(alpha):
		print("\n Analyzing metrics with min IoU of {}".format(min_iou))
		matches = metrics.pred_to_gt_assignment(all_ious, alpha=min_iou)
		M = [np.where(y>=min_iou, x, False) for (x,y) in zip(zero_matches,all_ious)] #this way, the matching doesn't change. I wonder how it compares to matches
		images_AP = metrics.calculate_average_precision(M, all_ious, all_intersections, all_unions, all_gt_mask_pix, all_pred_mask_pix)
		#a 1D array vector of length N images.
		all_image_AP.append(images_AP)
		all_matches.append(matches)

	qualities = metrics.get_matched_mask_qualities(all_matches, all_ious)

	#get CellPose metrics
	tp, fp, fn, ap = metrics.get_cellpose_precision(all_matches)

	#wrap items into a nice dictionary
	results = {}
	for variable in ["all_ious", "all_matches", "zero_matches", "all_image_AP", "qualities", "tp", "fp", "fn", "ap"]:
		results[variable] = eval(variable)

	return results


############################################################
"""Command Line"""
############################################################
#
if __name__ == '__main__':
	####################################################################
	"""UNCOMMENT FOR IMAGE AND MASK LOAD AND AUGMENT TEST"""
	####################################################################
	# print("Loading Cell Dataset")
	# cell_train = CellDataset()
	# config = CellConfig()
	# dataset_dir = os.path.join(ROOT_DIR,'datasets','cell')
	# cell_train.load_cell(dataset_dir,'train')
	# cell_train.prepare()
	# mask,_ = cell_train.load_mask(image_id=1)
	# image = cell_train.load_image(image_id=1,dimensionality="3D")
	# original_shape = image.shape
	# image, window, scale, padding, crop = utils.resize_image(
	#     image,
	#     min_dim=config.IMAGE_MIN_DIM,
	#     min_scale=config.IMAGE_MIN_SCALE,
	#     max_dim=config.IMAGE_MAX_DIM,
	#     mode=config.IMAGE_RESIZE_MODE)
	# mask = utils.resize_mask(mask, scale, padding, crop)
	# [image_out,mask_out]=test_augment(image,mask)
	# # molded_images, image_metas, windows = model.mold_inputs(images)


	#################################################################
	###############VIEW MODEL SUMMARY################
	#model.keras_model.summary()
	#############save model structure################
	#dot_img_file = '/users/czeddy/documents/model_3d_v2.png'
	#import tensorflow as tf
	#tf.keras.utils.plot_model(model.keras_model, to_file=dot_img_file, show_shapes=True)
	#################################################################

	###############ARGUMENTS PASSED TO FUNCTION######################
	import argparse

	# Parse command line arguments
	parser = argparse.ArgumentParser(
		description='Mask R-CNN for cell segmentation')
	parser.add_argument("command",
						metavar="<command>",
						help="'train', 'detect', or 'visualize'")
	parser.add_argument('--dataset', required=False,
						metavar="/path/to/dataset/",
						help='Root directory of the dataset')
	parser.add_argument('--weights', required=True,
						metavar="/path/to/weights.h5",
						help="Path to weights .h5 file or 'coco'")
	parser.add_argument('--logs', required=False,
						default=DEFAULT_LOGS_DIR,
						metavar="/path/to/logs/",
						help='Logs and checkpoints directory (default=logs/)')
	parser.add_argument('--subset', required=False,
						metavar="Dataset sub-directory",
						help="Subset of dataset to run prediction on")
	args = parser.parse_args()

	# Validate arguments
	if args.command == "train":
		assert args.dataset, "Argument --dataset is required for training"
	#elif args.command == "detect":
	#    assert args.subset, "Provide --subset to run prediction on"
	############################################################################
	if args.weights=="coco":
		args.weights=COCO_WEIGHTS_PATH
		#pass
	print("Weights: ", args.weights)
	if args.dataset:
		print("Dataset: ", args.dataset)
	if args.subset:
		print("Subset: ", args.subset)
	print("Logs: ", args.logs)

	# Configurations
	if args.command == "train":
		config = CellConfig()
	elif args.command == "detect":
		config = CellInferenceConfig()
	elif args.command == "metrics":
		config = CellInferenceConfig()
	else:
		print("using testing mode")
		config = CellInferenceConfig()
	config.display()

	# Create model
	#################################################################################
	"""UNCOMMENT OUT FOR THE STEM TEST MODEL"""
	#################################################################################
	# #3D model.
	# cell_train = CellDataset()
	# dataset_dir = os.path.join(ROOT_DIR,'datasets','cell')
	# cell_train.load_cell(dataset_dir,'train')
	# cell_train.prepare()
	#
	# model_3d = detect_with_stem_model(config,args.weights)
	# results_3d = detect(cell_train, model_3d, 0)
	# print('loading same model without stem branch')
	# model_2d = detect_without_stem_model(config,args.weights)
	# results_2d = detect(cell_train, model_2d, 0)
	#
	# image = cell_train.load_image(image_id=0,dimensionality="3D")
	# [outputs,m] = view_features(config, image, args, with_stem=True)

	#######################################################################
	if args.command == "train":
		model = modellib.MaskRCNN(mode="training", config=config,
								  model_dir=args.logs)
		load_weights(model.keras_model,args.weights)
		#model.keras_model.load_weights(args.weights, by_name=True)
		#model.keras_model.summary()
	elif args.command == "detect":
		if config.INPUT_DIM=="3D":
			model = detect_with_stem_model(config,args.weights)
		else:
			model = detect_without_stem_model(config,args.weights)
		#model.summary()
		#model = modellib.MaskRCNN(mode="inference", config=config,
		#                          model_dir=args.logs)

	#################################################################################
	"""UNCOMMENT OUT THE FOLLOWING TO TEST DATA LOADER FOR INFERENCE AND TRAINING"""
	#################################################################################
	# dataset_train = CellDataset()
	# # dataset_dir = os.path.join(ROOT_DIR,'datasets','cell')
	# dataset_train.load_cell(args.dataset, "train")
	# # dataset_train.load_cell(dataset_dir, "train")#, model.config.INPUT_DIM)
	# dataset_train.prepare()
	# # # [im_infer,infer_metas] = testing(model, dataset_train, args)
	# # #
	# # # #im should be the same as what is put into data generator. both in type and format
	# # #
	# [inp, op]=testing_train_generator(model, dataset_train)
	# import pdb;pdb.set_trace()
	# im_train = inp[0]
	# train_metas = inp[1]
	# im_masks = inp[6]#shape is 1x1024x1024x200 #where 200 is set by CellConfig as max instances.
	# im_mask_max = np.max(im_masks,axis=3)
	#
	# if config.INPUT_DIM=="3D":
	#     fig,((ax1,ax2,ax5),(ax3,ax4,ax6))=plt.subplots(2,3)
	#     ax1.imshow(im_train[0,:,:,3])
	#     ax2.imshow(im_train[0,:,:,4])
	#     train_max = np.max(im_train,axis=3)
	#     ax5.imshow(train_max[0])
	#     ax3.imshow(im_infer[0,:,:,3])
	#     ax4.imshow(im_infer[0,:,:,4])
	#     infer_max = np.max(im_infer,axis=3)
	#     ax6.imshow(infer_max[0])
	#     plt.show()
	#     fig,(ax1,ax2) = plt.subplots(1,2)
	#     ax1.imshow(train_max[0])
	#     ax2.imshow(im_mask_max[0])
	#     plt.show()
	# else:
	#     fig,(ax1,ax2)=plt.subplots(1,2)
	#     ax1.imshow(im_train[0])
	#     ax2.imshow(im_infer[0])
	#     plt.show()
	#     fig,(ax1,ax2)=plt.subplots(1,2)
	#     ax1.imshow(im)
	# #these all look good.
	# #the metas are different. In fact, the first number in the train generator meta is 7,
	# #whereas in the inference data it is zero. Why is that? What does that number represent? Pooling?
	# #no, that number SHOULD be the image ID number, which is random for the train generator.
	############################################################################################


	#Train or evaluate
	if args.command == "train":
		#pass
		train(model, args.dataset)
		# cell_test = CellDataset()
		# cell_test.load_cell(os.path.join(args.dataset,'train'))
		# cell_test.prepare()
		# pass
		#visualize.show_train_set(config, cell_test, n=1000)
	elif args.command == "detect":
		cell_test = CellDataset()
		cell_test.load_cell(args.dataset)
		cell_test.prepare()
		#import pdb;pdb.set_trace()
		# Create directory
		if not os.path.exists(RESULTS_DIR):
			os.makedirs(RESULTS_DIR)
		submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
		submit_dir = os.path.join(RESULTS_DIR, submit_dir)
		os.makedirs(submit_dir)

		detect(cell_test, model, config, submit_dir)
		#use the following line to run on a single image
		#detect(cell_test, model, config, submit_dir,image_id=22)

	elif args.command == "visualize":
		"""
		Use to visualize PEN output from Mask-RCNN
		"""
		image_id = 0
		assert config.INPUT_DIM=="3D"
		cell_test = CellDataset()
		cell_test.load_cell(args.dataset)
		cell_test.prepare()
		print("Running visualization on ", cell_test.image_info[image_id]['id'])
		image = cell_test.load_image(image_id,dimensionality="3D",z_to=config.INPUT_Z)
		#cents, zlocs, imshape = cell_test.load_z_positions(image_id)
		#need to pad image, if necessary.
		#########################################################################
		######################### PAD THE IMAGE #################################
		#########################################################################
		#pad the images
		if config.INPUT_DIM=="3D":
			image = cell_test.pad_z_image(image, z_to = config.INPUT_Z, center = True, random_pad = False)
		#########################################################################
		# [outputs,model] = view_features(config, image, args, with_stem=True)
		# #save these outputs in with the weights.
		# import tifffile
		# tifffile.imwrite(os.path.join(os.path.dirname(args.weights),cell_test.image_reference(0)+'_StemOut.tif'),outputs[0][0],photometric='rgb')
		# tifffile.imwrite(os.path.join(os.path.dirname(args.weights),cell_test.image_reference(0)+'_StemOut_R.tif'),outputs[0][0,:,:,0])
		# tifffile.imwrite(os.path.join(os.path.dirname(args.weights),cell_test.image_reference(0)+'_StemOut_G.tif'),outputs[0][0,:,:,1])
		# tifffile.imwrite(os.path.join(os.path.dirname(args.weights),cell_test.image_reference(0)+'_StemOut_B.tif'),outputs[0][0,:,:,2])
		#import pdb;pdb.set_trace()
		[outputs,M] = view_stem_output(image, config, args)
		#save these outputs in with the weights.
		import tifffile
		tifffile.imwrite(os.path.join(os.path.dirname(args.weights),cell_test.image_reference(0)+'_StemOut.tif'),outputs,photometric='rgb')
		tifffile.imwrite(os.path.join(os.path.dirname(args.weights),cell_test.image_reference(0)+'_StemOut_R.tif'),outputs[:,:,0])
		tifffile.imwrite(os.path.join(os.path.dirname(args.weights),cell_test.image_reference(0)+'_StemOut_G.tif'),outputs[:,:,1])
		tifffile.imwrite(os.path.join(os.path.dirname(args.weights),cell_test.image_reference(0)+'_StemOut_B.tif'),outputs[:,:,2])
		print("Saving PEN output to ", os.path.dirname(args.weights))

	elif args.command == "metrics":
		cell_test = CellDataset()
		cell_test.load_cell(args.dataset)
		cell_test.prepare()
		results = measure_metrics(cell_test, config, alpha=list(np.linspace(0.5,0.95, int((0.95-0.5)/0.05)+2)))
		print("You should use np.savez_compressed and save the metrics.")
		#np.savez_compressed("MaskRCNN_metrics.npz", **results)
	else:
		print("'{}' is not recognized. "
			  "Use 'train' or 'detect' or 'visualize'".format(args.command))
