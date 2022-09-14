"""
Code to put here includes dynamics, create flow maps
Import binary ground truth masks.
Use centroid or median pixel locations as centers.
Do heat map for each object.
"""
import sys
import os
import logging
import math
import random
import numpy as np
import scipy
import scipy.signal
from scipy.ndimage import convolve
import skimage.color
import skimage.io
import skimage.transform
from skimage.morphology import erosion, square
import json
import matplotlib.pyplot as plt
from distutils.version import LooseVersion
import warnings
import time
import sklearn.decomposition
import copy
################################################################################
################################################################################
################################################################################
################################################################################

def load_json_data(pth):
	with open(pth) as f:
		data=json.load(f)
	return data

class Dataset(object):
	"""The base class for dataset classes.
	To use it, create a new class that adds functions specific to the dataset
	you want to use. For example:
	class CatsAndDogsDataset(Dataset):
		def load_cats_and_dogs(self):
			...
		def load_mask(self, image_id):
			...
		def image_reference(self, image_id):
			...
	See COCODataset and ShapesDataset as examples.
	"""

	def __init__(self, class_map=None):
		self._image_ids = []
		self.image_info = []
		# Background is always the first class
		self.class_info = [{"source": "", "id": 0, "name": "BG"}]
		self.source_class_ids = {}

	def add_class(self, source, class_id, class_name):
		assert "." not in source, "Source name cannot contain a dot"
		# Does the class exist already?
		for info in self.class_info:
			if info['source'] == source and info["id"] == class_id:
				# source.class_id combination already available, skip
				return
		# Add the class
		self.class_info.append({
			"source": source,
			"id": class_id,
			"name": class_name,
		})

	def add_image(self, source, image_id, path, **kwargs):
		image_info = {
			"id": image_id,
			"source": source,
			"path": path,
		}
		image_info.update(kwargs)
		self.image_info.append(image_info)

	def image_reference(self, image_id):
		"""Return a link to the image in its source Website or details about
		the image that help looking it up or debugging it.
		Override for your dataset, but pass to this function
		if you encounter images not in your dataset.
		"""
		return ""

	def prepare(self, class_map=None):
		"""Prepares the Dataset class for use.
		TODO: class map is not supported yet. When done, it should handle mapping
			  classes from different datasets to the same class ID.
		"""

		def clean_name(name):
			"""Returns a shorter version of object names for cleaner display."""
			return ",".join(name.split(",")[:1])

		# Build (or rebuild) everything else from the info dicts.
		self.num_classes = len(self.class_info)
		self.class_ids = np.arange(self.num_classes)
		self.class_names = [clean_name(c["name"]) for c in self.class_info]
		self.num_images = len(self.image_info)
		self._image_ids = np.arange(self.num_images)

		# Mapping from source class and image IDs to internal IDs
		self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
									  for info, id in zip(self.class_info, self.class_ids)}
		self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
									  for info, id in zip(self.image_info, self.image_ids)}

		# Map sources to class_ids they support
		self.sources = list(set([i['source'] for i in self.class_info]))
		self.source_class_ids = {}
		# Loop over datasets
		for source in self.sources:
			self.source_class_ids[source] = []
			# Find classes that belong to this dataset
			for i, info in enumerate(self.class_info):
				# Include BG class in all datasets
				if i == 0 or source == info['source']:
					self.source_class_ids[source].append(i)

	def map_source_class_id(self, source_class_id):
		"""Takes a source class ID and returns the int class ID assigned to it.
		For example:
		dataset.map_source_class_id("coco.12") -> 23
		"""
		return self.class_from_source_map[source_class_id]

	def get_source_class_id(self, class_id, source):
		"""Map an internal class ID to the corresponding class ID in the source dataset."""
		info = self.class_info[class_id]
		assert info['source'] == source
		return info['id']

	@property
	def image_ids(self):
		return self._image_ids

	def source_image_link(self, image_id):
		"""Returns the path or URL to the image.
		Override this to return a URL to the image if it's available online for easy
		debugging.
		"""
		return self.image_info[image_id]["path"]

	def load_image(self, image_id):
		"""Load the specified image and return a [H,W,3] Numpy array.
		"""
		# Load image
		image = skimage.io.imread(self.image_info[image_id]['path'])
		# If grayscale. Convert to RGB for consistency.
		if image.ndim != 3:
			image = skimage.color.gray2rgb(image)
		# If has an alpha channel, remove it for consistency
		if image.shape[-1] == 4:
			image = image[..., :3]
		return image

	def load_mask(self, image_id):
		"""Load instance masks for the given image.
		Different datasets use different ways to store masks. Override this
		method to load instance masks and return them in the form of am
		array of binary masks of shape [height, width, instances].
		Returns:
			masks: A bool array of shape [height, width, instance count] with
				a binary mask per instance.
			class_ids: a 1D array of class IDs of the instance masks.
		"""
		# Override this function to load a mask from your dataset.
		# Otherwise, it returns an empty mask.
		logging.warning("You are using the default load_mask(), maybe you need to define your own one.")
		mask = np.empty([0, 0, 0])
		class_ids = np.empty([0], np.int32)
		return mask, class_ids

################################################################################
################################################################################
################################################################################
################################################################################

class CellDataset(Dataset):

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
			RR, CC = skimage.draw.polygon_perimeter(poly[:,0], poly[:,1], mask.shape[0:-1])
			try:
				mask[rr,cc,i] = 1
			except:
				print("too many objects, needs debugging")
				print(self.image_info[image_id])

			#put each annotation in a different channel.

		# Return mask, and array of class IDs of each instance. Since we have
		# one class ID only, we return an array of 1s
		return mask.astype(np.bool)

	def load_detection(self, image_id):
		"""Generate data from detections!
	    Returns:
		masks: A float array of shape [height, width, N output channels] with
			range 0-1, predictions of cell vs background
		xgrad: A float array of shape [height, width, N output channels] with
			Predictions of x-gradients
		ygrad: A float array of shape [height, width, N output channels] with
			Predictions of y-gradients
		edges: A float array of shape [height, width, N output channels] with
			range 0-1, predictions of cell border vs background.

		"""
		# If not a balloon dataset image, delegate to parent class.
		image_info = self.image_info[image_id]
		if image_info["source"] != "cell":
			return super(self.__class__, self).load_mask(image_id)
			#see config.py for parent class default load_mask function

		# Get mask directory from image path
		if not hasattr(self, 'detect_dir'):
			self.detect_dir = os.path.join(image_info['path'], "detections")
		#detect_dir = os.path.join(image_info['path'], "detections")
		#os.path.join(os.path.dirname(os.path.dirname(image_info['path'])), "gt")
		#check if _results file exists
		if os.path.isfile(os.path.join(self.detect_dir,image_info['id']+"_results.npz")):
			#load object detection masks, put each in its own channel.
			#make the edges data all zeros so there is no border to remove.
			#also, in this case, we don't want to remove the borders.
			#so return a flag of some kind too.
			data = np.load(os.path.join(self.detect_dir,image_info['id']+"_results.npz"))
			masks = data['label']
			(h,w,N) = masks.shape
			#currently, masks are labeled structures. To be congruent with the rest of the code,
			#instead, we should put every detection in its own channel.
			#import pdb;pdb.set_trace()
			n_detect = np.max(masks)#np.sum([np.max(masks[:,:,ch]) for ch in range(N)])
			#masks are already ordered.
			out_masks = np.zeros(shape=(h,w,int(n_detect)),dtype=np.bool)
			counter = 0
			for ch in range(masks.shape[2]):
				for obj in np.unique(masks[:,:,ch])[1:]:#range(1,int(np.max(masks[:,:,ch]))): #skip background
					out_masks[:,:,counter]=np.where(masks[:,:,ch]==obj,True,False)
					counter+=1
			border = np.zeros(shape=out_masks.shape,dtype=np.bool)
			return_flag = True
			return out_masks, border, return_flag
		else:
			data = np.load(os.path.join(self.detect_dir,image_info['id']+".npz")) #load file with same name.
			return_flag = False
			return data['mask'], data['edges'], return_flag


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


	def load_image(self, image_id, dimensionality, mask=None, avg_pixel=None):
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
			image = image.astype(np.float32)
			#Note: we are doing a per image centering, that is every image should have a mean of zero, although it isn't really since the negative values are thresholded.

			#sometimes images are loaded with range 0-1 rather than 0-255.
			if np.max(image)<=1.:
				image = image*255.0
				#we'll return to 0-1 range at the end.

			if avg_pixel is None:
				mean_val = scipy.stats.tmean(image.ravel(),(0,100))
				image = image - mean_val
				image[image<0]=0
			else:
				image = image - avg_pixel
				image[image<0]=0

			# If grayscale. Convert to RGB for consistency.
			if image.ndim != 3:
				image = skimage.color.gray2rgb(image)
			# If has an alpha channel, remove it for consistency
			# if image.shape[-1] == 4:
			# 	image = image[..., :3]

			#the only time load_image is not passed with a mask argument, we are in inference mode. Otherwise, always training.
			#in inference, we don't have a "weight map". CE 12/15/21
			if mask is not None:
				bad_pixels = self.load_weight_map(image_id)
				#now
				#import pdb; pdb.set_trace()
				#if mask is not None:
				mask = np.max(mask,axis=-1) #take max projection
				mask = np.expand_dims(mask,axis=-1) #add dimension for np.where
				bad_pixels=np.where(mask==True,False,bad_pixels) #anywhere an annotated object is, we don't want to cancel it ou

				#for each channel in image, set these to the mode of image.
				#determine the mean of small numbers.
				image = np.where(bad_pixels==True, 0.0, image)
				#image output shape is [H,W,3]
				#image = image / np.max(image)
		else:
			"""Load the specified image and return a [H,W,Z,1] Numpy array.
			"""

			#print(os.path.join(self.image_info[image_id]['path'],'images',self.image_info[image_id]['id']+'.ome.tif'))
			#we'll fix z
			#z_to = 15
			#ultimately, we'll do enough convolutions to get that down to the correct size.
			image = skimage.io.imread(os.path.join(self.image_info[image_id]['path'],'images',self.image_info[image_id]['id']+'.ome.tif'))

			##making new aray and filling it is faster than using pad, but only if we use "zeros" and not "full".
			##for a nonzero padding value, it is slower this way.
			image = image.astype(np.float32)

			#sometimes images are loaded with range 0-1 rather than 0-255.
			if np.max(image)<=1.:
				image = image*255.0
				#again, we will return to a 0-1 range at the end.

			if avg_pixel is None:
				pad_val = scipy.stats.tmean(image.ravel(),(0,100)) #notice we are excluding the cell objects.
				image = image - pad_val
				image[image<0]=0 #clip values. #this clip values was at 1 before.
			else:
				image = image - avg_pixel
				image[image<0]=0

			#sometimes images load as H x W x Z, sometimes by Z x H x W. we need latter
			if len(image.shape)==2:
				image = np.expand_dims(image, axis = 0)

			if image.shape[2] < image.shape[0]:
				print("The shape of input is H x W x Z rather than Z x H x W")

			#roll axis.
			image = np.rollaxis(image, 0, 3)

			"""
			Removed padding at this step and placed in load_image_gt and load_image_inference
			"""

			#see previous 2D version CE 12/15/21
			if mask is not None:
				#load weight map
				bad_pixels = self.load_weight_map(image_id)
				#if mask is not None:
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
			#image = image / np.max(image)
			#should currently be between the range of 0-255, conver to 0-1
			#image /= 255.
			#Already float32 dtype.
		return image/255.

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

	def load_z_class(self, image_id, num_k, zsize=None, flip=False):
		# If not a balloon dataset image, delegate to parent class.
		image_info = self.image_info[image_id]
		#if image_info["source"] != "cell":
		#	return super(self.__class__, self).load_mask(image_id)
		#	#see config.py for parent class default load_mask function

		# Get mask directory from image path
		json_dir = os.path.join(image_info['path'], "gt")
		#os.path.join(os.path.dirname(os.path.dirname(image_info['path'])), "gt")
		data = load_json_data(os.path.join(json_dir,image_info['id']+".json"))
		zlocs = np.array(data['annotations']['regions']['approx_z'])
		if flip:
			data_width = data['images']['slices']
			zlocs = (data_width-1) - zlocs
		#Now, we need to determine classification.
		zclasses = self.run_k_means(zlocs, num_k, num_it=300)
		# see cellpose20220115T1130 for N=3 output channel
		# see cellpose20220116T for N=5 output channel
		#zclasses = self.auto_slice_individual_stack(zlocs, num_k) ##see cellpose20220116T1425
		#zclasses = self.auto_slice_whole_stack(zlocs, num_k, zsize) #see cellpose20220119T0909

		return zclasses

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


	def load_z_class_alt(self, image_id, num_k, zmethod, cents=None, zlocs=None, zsize=None, zflip=False, yflip=False, xflip=False, xyflip=False):
		"""
		zmethod is dictionary passed from config.
		xyflip if rot90 was used. Should come AFTER doing the x and y flips.
		"""
		# If not a balloon dataset image, delegate to parent class.
		image_info = self.image_info[image_id]
		#if image_info["source"] != "cell":
		#	return super(self.__class__, self).load_mask(image_id)
		#	#see config.py for parent class default load_mask function

		# Get mask directory from image path
		json_dir = os.path.join(image_info['path'], "gt")
		#os.path.join(os.path.dirname(os.path.dirname(image_info['path'])), "gt")
		data = load_json_data(os.path.join(json_dir,image_info['id']+".json"))

		image_z_size = data['images']['slices']

		zlocs = np.array(data['annotations']['regions']['approx_z'])

		cents = data['annotations']['regions']['centroid']
		imsize = [data['images']['height'], data['images']['width']]

		if zflip:
			zlocs = (image_z_size-1) - zlocs

		# zlocs = np.concatenate([zlocs+(n*image_z_size) for n in range(n_stacked)])
		#
		# if made_dense:
		# 	#in this case, we copied the image onto itself, z-flipped and rotated it.
		# 	#right now, we don't have the record of how many times we rotated it,
		# 	#although that could be a passed argument for pca_kmeans.
		# 	#made_dense also occurs AFTER we stack, if we stack.
		# 	zlocs = np.concatenate([zlocs, (image_z_size-1) - zlocs])

		if zmethod['pca_kmeans']:
			#use x-y-z location to do PCA to transform data.
			if xflip:
				cents = [[imsize[0]-1-x[0],x[1]] for x in cents]
			if yflip:
				cents = [[x[0],imsize[1]-1-x[1]] for x in cents]
			if xyflip:
				cents = [[imsize[1]-1-x[1],x[0]] for x in cents]
			cents = np.concatenate([np.array(cents)]*n_stacked)
			locs = np.array([[x[0],x[1],z] for x,z in zip(cents,zlocs)])
			#standardize the data first.
			locssd = (locs - np.mean(locs,axis=0)) / np.std(locs,axis=0)
			#use PCA
			pca = sklearn.decomposition.PCA(n_components=3)
			#transform locs to PCA space.
			transformedLocs = pca.fit_transform(locssd)
			pc2 = transformedLocs[:,2]
			#Now, we need to determine classification.
			zclasses = self.run_k_means(pc2, num_k, num_it=300)

		elif zmethod['z_kmeans']:
			zclasses = self.run_k_means(zlocs, num_k, num_it=300)

		elif zmethod['slice_cells']:
			zclasses = self.auto_slice_individual_stack(zlocs, num_k) ##see cellpose20220116T1425

		elif zmethod['slice_stack']:
			zclasses = self.auto_slice_whole_stack(zlocs, num_k, zsize) #see cellpose20220119T0909


		return zclasses

	def load_z_class_alt2(self, cents, zlocs, imshape, num_k, zmethod, zsize=None, zflip=False, yflip=False, xflip=False, xyflip=False):
		"""
		zmethod is dictionary passed from config.
		xyflip if rot90 was used. Should come AFTER doing the x and y flips.
		"""
		image_z_size = imshape[2]

		imsize = [imshape[0], imshape[1]]

		if zflip:
			zlocs = (image_z_size-1) - zlocs

		if zmethod['pca_kmeans']:
			#use x-y-z location to do PCA to transform data.
			if xflip:
				cents = [[imsize[0]-1-x[0],x[1]] for x in cents]
			if yflip:
				cents = [[x[0],imsize[1]-1-x[1]] for x in cents]
			if xyflip:
				cents = [[imsize[1]-1-x[1],x[0]] for x in cents]
			cents = np.array(cents)
			locs = np.array([[x[0],x[1],z] for x,z in zip(cents,zlocs)])
			#standardize the data first.
			locssd = (locs - np.mean(locs,axis=0)) / np.std(locs,axis=0)
			#use PCA
			pca = sklearn.decomposition.PCA(n_components=3)
			#transform locs to PCA space.
			transformedLocs = pca.fit_transform(locssd)
			pc2 = transformedLocs[:,2]
			#Now, we need to determine classification.
			zclasses = self.run_k_means(pc2, num_k, num_it=300)

		elif zmethod['z_kmeans']:
			zclasses = self.run_k_means(zlocs, num_k, num_it=300)

		elif zmethod['slice_cells']:
			zclasses = self.auto_slice_individual_stack(zlocs, num_k) ##see cellpose20220116T1425

		elif zmethod['slice_stack']:
			zclasses = self.auto_slice_whole_stack(zlocs, num_k, zsize) #see cellpose20220119T0909

		elif zmethod['random']:
			#randomly assign cells to range of 0-num_k-1
			zclasses = np.random.randint(0, num_k, size=len(zlocs))


		return zclasses



	def run_k_means(self, data, num_k, num_it = 30):
		"""
		data should be z-score normalized.
		data should be an N x d array, where d are features.
		Initializes centers equal distance through Z-stack.
		"""
		assert len(data.shape)==1, "Data must be 1-D"

		if len(np.unique(data))<=num_k:
			"""
			Here, there are less slices with cells than there are channels available.
			Instead, we place each unique slice in its own channel.
			"""
			zs = np.unique(data)
			zclasses = list(range(len(zs)))
			zclasses = np.array([zclasses[i] for i,z in enumerate(zs) for thisz in data if thisz==z])
			return zclasses
		else:
			"""
			Here, there are more slices with cells than there are channels available.
			We will run a k-means to determine optimal slice selection.
			"""
			#import pdb;pdb.set_trace()
			#normalize the data.
			#data -= np.mean(data)
			#data /= np.std(data)

			#It is 1-D, but at we'll expand it to 2D for data purposes
			if len(data.shape)==1:
				data = np.expand_dims(data,axis=1)

			start = np.min(data)
			end = np.max(data)
			centers = np.expand_dims(np.linspace(start, end, num_k),axis=-1)
			#there are other options than linspace
			#this should be reproducible since we aren't randomly setting the starting points.
			for it in range(num_it):
				#calculate distance from points in data to centers. Should be a N x k array.
				dists = np.zeros((data.shape[0],num_k))
				for ci in range(num_k):
					dists[:,ci] = np.sqrt(np.sum(np.square(data - centers[ci,:]),axis=1))
				#assign cell to cluster based on furthest distance.
				assigned_clusters = np.argmin(dists,1) #size N array
				#recalculate mean centers.
				for ci in np.unique(assigned_clusters):#range(k):
					centers[ci] = np.mean(data[assigned_clusters==ci,:],0)

			return assigned_clusters

	def auto_slice_individual_stack(self, data, num_k):
		"""
		Z-information
		Trying to teach machine to ignore the zero padding in the 3D stack.
		SPLITS the individual image stack (ex. stack has 8 slices) into num_k groups.
		"""
		assert len(data.shape)==1, "Data must be 1-D"

		"""
		Here, there are more slices with cells than there are channels available.
		We will seperate the cells by slice location only.
		"""
		#find minimum and maximum slice locations.
		zmin = int(np.min(data))
		zmax = int(np.max(data))
		slices = list(range(zmin,zmax+1,1))
		s1 = int(np.round(len(slices)/num_k)) #width of N-1 stacks.
		s2 = len(slices)%((num_k-1)*s1) #width of final stack
		#determine corresponding output channels for each slice.
		slice_channel = [[n]*s1 if n<num_k-1 else [n]*s2 for n in range(num_k)]
		slice_channel = [n for x in slice_channel for n in x]
		#assign cell location slice to channel.
		zs = np.unique(data)
		zclasses = np.array([slice_channel[i] for i,slice in enumerate(slices) for thisz in data if thisz==slice])
		return zclasses

	def auto_slice_whole_stack(self, data, num_k, zsize):
		"""
		Z-information, most naive attempt
		"""
		assert len(data.shape)==1, "Data must be 1-D"

		"""
		Using config.INPUT_Z as zsize, splits every stack into channels of size
		config.INPUT_Z / config.OUT_CHANNELS
		"""
		#find minimum and maximum slice locations.
		zmin = 0
		zmax = zsize-1
		slices = list(range(zmin,zmax+1,1))
		s1 = int(np.round(len(slices)/num_k)) #width of N-1 stacks.
		s2 = len(slices)%((num_k-1)*s1) #width of final stack
		#determine corresponding output channels for each slice.
		slice_channel = [[n]*s1 if n<num_k-1 else [n]*s2 for n in range(num_k)]
		slice_channel = [n for x in slice_channel for n in x]
		#assign cell location slice to channel.
		zs = np.unique(data)
		zclasses = np.array([slice_channel[i] for i,slice in enumerate(slices) for thisz in data if thisz==slice])
		return zclasses

	def run_pca(self, data, num_k, num_it = 30):
		"""
		data should be x-y-z location.
		Run PCA on x-y-z locations of cells. Then take the second principle
		component data and feed into run_k_means.
		"""

	def check_z_dist(self):
		# If not a balloon dataset image, delegate to parent class.
		zlocs_rel = []
		zlocs = []
		for image_id in range(len(self.image_info)):
			image_info = self.image_info[image_id]
			# Get mask directory from image path
			json_dir = os.path.join(image_info['path'], "gt")
			#os.path.join(os.path.dirname(os.path.dirname(image_info['path'])), "gt")
			data = load_json_data(os.path.join(json_dir,image_info['id']+".json"))
			thisz = data['annotations']['regions']['approx_z']
			if isinstance(thisz, int):
				thisz = [thisz]
			zlocs.append(thisz)
			data_width = data['images']['slices']
			zlocs_rel.append([z/data_width for z in thisz])
		zlocs = [z for x in zlocs for z in x ] #assuming each 'thisz' is a list.
		zlocs_rel = [z for x in zlocs_rel for z in x]
		#return zlocs, zlocs_rel

		#Now, we need to determine classification.
		zclasses = self.run_k_means(zlocs, num_k, num_it=300)
		# see cellpose20220115T1130 for N=3 output channel
		# see cellpose20220116T for N=5 output channel
		#zclasses = self.auto_slice_individual_stack(zlocs, num_k) ##see cellpose20220116T1425
		#zclasses = self.auto_slice_whole_stack(zlocs, num_k, zsize) #see cellpose20220119T0909

		return zclasses


	def image_reference(self, image_id):
		"""Return the path of the image."""
		info = self.image_info[image_id]
		if info["source"] == "cell":
			return info["id"]
		else:
			super(self.__class__, self).image_reference(image_id)


#################################

class CellPoseDataset(Dataset):
	"""
	Dataset from MouseVision
	Consists of PNG formatted images and PNG labeled masks
	Filename format follows
	NAME_img.png for IMAGE
	NAME_mask.png for MASK
	"""

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

		image_ids = os.listdir(load_dir)
		#we should further filter these if they are "." files.
		image_ids = [x for x in image_ids if x[0]!="."]

		#image_ids = next(os.walk(load_dir))[2] #this returns directory names inside dataset_dir.
		#sort image_ids
		image_ids.sort()

		# Add images
		for image_id in image_ids:
			if subset:
				self.add_image(
					"cell",
					image_id=image_id[0:image_id.find("_")],
					path=os.path.join(dataset_dir,subset))#,"images",image_id))#os.path.join(dataset_dir, image_id, "images/{}.png".format(image_id)))
			else:
				self.add_image(
					"cell",
					image_id=image_id[0:image_id.find("_")],
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

		m=skimage.io.imread(os.path.join(mask_dir,image_info['id']+"_masks.png"))
		#this is a labeled image.
		masks=[np.where(m==i+1,1,0) for i in range(int(np.max(m))) if np.sum(m==i+1)>25]
		#Note, this threshold exists because MouseVision has accidentally included some mislabels (i.e. 061_mask, object 132)
		masks = np.stack(masks)
		masks = np.rollaxis(masks,0,3)
		# Return mask, and array of class IDs of each instance. Since we have
		# one class ID only, we return an array of 1s
		return masks.astype(np.bool_) #, np.ones([mask.shape[-1]], dtype=np.int32)

	def load_image(self, image_id):
		image = skimage.io.imread(os.path.join(self.image_info[image_id]['path'],'images',self.image_info[image_id]['id']+'_img.png'))
		# If grayscale. Convert to RGB for consistency.
		if image.ndim != 3:
			image = skimage.color.gray2rgb(image)
		# If has an alpha channel, remove it for consistency
		if image.shape[-1] == 4:
			image = image[..., :3]

		image = image.astype(np.float32)
		mean_val = scipy.stats.tmean(image.ravel(),(0,100))
		image = image - mean_val
		image[image<0]=0
		#image output shape is [H,W,3]
		image = image / np.max(image)
		#return as HxWx3
		image = np.max(image,axis=2)
		image = np.stack((image,image,image),axis=2)

		return image

	def image_reference(self, image_id):
		"""Return the path of the image."""
		info = self.image_info[image_id]
		if info["source"] == "cell":
			return info["id"]
		else:
			super(self.__class__, self).image_reference(image_id)



##################load images

def load_json_data(pth):
	with open(pth) as f:
		data=json.load(f)
	return data

def load_image_inference(dataset, config, image_id, max_size=1024, min_size=None, z_begin=None):
	"""Load and return ground truth data for an image (image, mask, xgrad, ygrad).

	Returns:
	image: [height, width, 3] for 2D, [height, width, Z, 1] for 3D
	"""
	# Load image and mask and gradients

	#here, we actually want z_to to be whatever the image is actually at...
	#No, that is incorrect. The stem model is made so the stride lengths are appropriate for the set size.
	if hasattr(config,"INPUT_DIM"):
		"""
		Dataset is from CellDataset
		"""
		if config.INPUT_DIM=="3D":
			image = dataset.load_image(image_id, config.INPUT_DIM, avg_pixel=config.AVG_PIX) #shape is HxWxZ
			#pad image
			if z_begin is None:
				image = dataset.pad_z_image(image, z_to = config.INPUT_Z, center=True, random_pad=False) #do center padding.
			else:
				image = dataset.pad_z_image(image, z_to = config.INPUT_Z, z_begin=z_begin, center=config.Padding_Opts['center'], random_pad=config.Padding_Opts['random'])
		#else:
		#	image = dataset.load_image(image_id, config.INPUT_DIM, avg_pixel=config.AVG_PIX)
		#	#image is 2D.
		else:
			#image input is 2D - without PEN.
			if config.project_mode=="max":
				image = dataset.load_image(image_id, config.INPUT_DIM, avg_pixel=config.AVG_PIX)
			else:
				#mode is "linear". Encode depth. During training, the image is
				#NOT padded before.
				"""
				By not padding, we allow the linear model to stretch over the unique image.
				The feature of this is better segmentation. The drawback is an inconsistent
				shading/depth ratio.
				"""
				image = dataset.load_image(image_id, "3D", avg_pixel=config.AVG_PIX) #shape is HxWxZx1

				if config.pad_before_linear:
					#pad the images FIRST. See config settings.
					if z_begin is None:
						image = dataset.pad_z_image(image, z_to = config.INPUT_Z, center=True, random_pad=False) #do center padding.
					else:
						image = dataset.pad_z_image(image, z_to = config.INPUT_Z, z_begin=z_begin, center=config.Padding_Opts['center'], random_pad=config.Padding_Opts['random'])

				image = linearly_color_encode_image(image)
	else:
		"""
		Dataset is from CellPoseDataset
		"""
		image = dataset.load_image(image_id)

	#Added CE 06/09/22
	if min_size is not None:
		assert min_size%2**(config.UNET_DEPTH-1)==0, "'min_size' argument of {} is not an appropriate size for specified UNET depth".format(min_size)
		h,w = image.shape[:2]
		if any([True if x<min_size else False for x in image.shape[:2]]):
			h_diff = max(0,min_size - h)
			w_diff = max(0,min_size - w)
			image = np.pad(image,tuple([(math.floor(h_diff/2),math.ceil(h_diff/2)),
					(math.floor(w_diff/2),math.ceil(w_diff/2))]+[(0,0) for i in range(image.ndim - 2)]),
					constant_values=tuple([(0,0) for i in range(image.ndim)]))
			window = (math.floor(h_diff/2),h+math.ceil(h_diff/2),math.floor(w_diff/2),w+math.ceil(w_diff/2))
		else:
			window=(0,w,0,h)

	else:
		#we need to verify that the image is of the right shape.
		#based on the strides, it needs to be able to be divisible by 2 at least UNET_DEPTH-1 times
		h,w = image.shape[:2]

		if any([True if x<max_size else False for x in image.shape[:2]]):
			if h / 2**(config.UNET_DEPTH-1) != int(h / 2**(config.UNET_DEPTH-1)):
				pad_height=True
			else:
				pad_height=False
			if w / 2**(config.UNET_DEPTH-1) != int(w / 2**(config.UNET_DEPTH-1)):
				pad_width=True
			else:
				pad_width=False
			if any([pad_height,pad_width]):
				print("Inference image requires padding to be able to be analyzed. Padding to closest shapes...")
				choices=[i for i in range(max_size+1) if i%2**(config.UNET_DEPTH-1)==0]
				h_padded = choices[min([i for i,x in enumerate(choices) if x>h])]
				w_padded = choices[min([i for i,x in enumerate(choices) if x>w])]
				h_diff = h_padded-h
				w_diff = w_padded-w
				# image = np.pad(image,((math.floor(h_diff/2),math.ceil(h_diff/2)),
				# 		(math.floor(w_diff/2),math.ceil(w_diff/2)),(0,0)),
				# 		constant_values=((0,0),(0,0),(0,0)))
				image = np.pad(image,tuple([(math.floor(h_diff/2),math.ceil(h_diff/2)),
				 		(math.floor(w_diff/2),math.ceil(w_diff/2))]+[(0,0) for i in range(image.ndim - 2)]),
						constant_values=tuple([(0,0) for i in range(image.ndim)]))
				window = (math.floor(h_diff/2),h+math.ceil(h_diff/2),math.floor(w_diff/2),w+math.ceil(w_diff/2))
			else:
				window=(0,w,0,h)
		else:
			window=(0,w,0,h)

	return image, window

def load_image_gt_metrics(dataset, config, image_id):
	"""Load and return ground truth data for an image (image, mask, xgrad, ygrad).
	augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
		For example, passing imgaug.augmenters.Fliplr(0.5) flips images
		right/left 50% of the time.
	augmentation should be passed as a DICTIONARY, from config.
	Returns:
	image: [height, width, 3] for 2D, [height, width, Z, 1] for 3D
	mask: [height, width]
	xgrad: [height, width]
	ygrad: [height, width]

	"""

	mask = dataset.load_mask(image_id)

	borders = create_borders_N(mask)

	# # Load image and mask and gradients
	# if hasattr(config,"INPUT_DIM"):
	# 	"""
	# 	Dataset is from CellDataset
	# 	"""
	# 	if config.INPUT_DIM=="3D":
	# 		image = dataset.load_image(image_id, config.INPUT_DIM, mask=mask) #shape is HxWxZ
	# 		#pad the images
	# 		image = dataset.pad_z_image(image, z_to = config.INPUT_Z, center = config.Padding_Opts['center'], random_pad = config.Padding_Opts['random'])
	# 	else:
	# 		image = dataset.load_image(image_id, config.INPUT_DIM, mask=mask)
	# 		#image is 2D
	# else:
	# 	"""
	# 	Dataset is from CellPoseDataset
	# 	"""
	# 	image = dataset.load_image(image_id)
	#
	#
	# if config.INPUT_DIM=="3D":
	# 	zclasses = dataset.load_z_class_alt(image_id, config.OUT_CHANNELS,
	# 	 			config.GT_ASSIGN, config.INPUT_Z)
	# else:
	# 	zclasses = dataset.load_z_class_alt(image_id, 1, config.GT_ASSIGN)

	"""
	Replaced above with a single function that places objects simultaneously in random channels.
	"""
	#mask, borders, xgrad, ygrad = _project_all_by_Z(mask, zclasses, config.OUT_CHANNELS)

	"""
	Pull predictions!
	"""
	pred_masks, pred_borders, return_flag = dataset.load_detection(image_id)
	if not return_flag:
		print("Using unprocessed file! Expect poor precision.")

	#return mask, xgrad, ygrad, borders, pred_masks, pred_x, pred_y, pred_borders
	return mask, borders, pred_masks, pred_borders, return_flag

def load_image_PadCheck(dataset, config, image_id, max_size=1024, z_before=0):
	"""Load and return ground truth data for an image (image, mask, xgrad, ygrad).
	Loads image and pads in axial dimension based on argument z_before
	Returns:
	image: [height, width, 3] for 2D, [height, width, Z, 1] for 3D
	"""
	# Load image and mask and gradients

	#here, we actually want z_to to be whatever the image is actually at...
	#No, that is incorrect. The stem model is made so the stride lengths are appropriate for the set size.
	assert hasattr(config,"INPUT_DIM"), "Data must be set for 3D"
	"""
	Dataset is from CellDataset
	"""
	assert config.INPUT_DIM=="3D", "image must be 3D."
	image_og = dataset.load_image(image_id, config.INPUT_DIM, avg_pixel=config.AVG_PIX) #shape is HxWxZ
	z_size = image_og.shape[2]
	#pad image
	image = dataset.pad_z_image(image_og, z_to = config.INPUT_Z, z_begin=z_before) #pad based on z_before argument.


	#we need to verify that the image is of the right shape.
	#based on the strides, it needs to be able to be divisible by 2 at least UNET_DEPTH-1 times
	h,w = image.shape[:2]

	if any([True if x<max_size else False for x in image.shape[:2]]):
		if h / 2**(config.UNET_DEPTH-1) != int(h / 2**(config.UNET_DEPTH-1)):
			pad_height=True
		else:
			pad_height=False
		if w / 2**(config.UNET_DEPTH-1) != int(w / 2**(config.UNET_DEPTH-1)):
			pad_width=True
		else:
			pad_width=False
		if any([pad_height,pad_width]):
			print("Inference image requires padding to be able to be analyzed. Padding to closest shapes...")
			choices=[i for i in range(max_size+1) if i%2**(config.UNET_DEPTH-1)==0]
			h_padded = choices[min([i for i,x in enumerate(choices) if x>h])]
			w_padded = choices[min([i for i,x in enumerate(choices) if x>w])]
			h_diff = h_padded-h
			w_diff = w_padded-w
			image = np.pad(image,((math.floor(h_diff/2),math.ceil(h_diff/2)),
					(math.floor(w_diff/2),math.ceil(w_diff/2)),(0,0)),
					constant_values=((0,0),(0,0),(0,0)))
	return image

def crop_center(img,cropx,cropy):
	y,x = img.shape[0:2]
	startx = x//2-(cropx//2)
	starty = y//2-(cropy//2)
	return img[starty:starty+cropy,startx:startx+cropx]

def create_borders(binary_mask):
	"""Given mask, pull out vertices
	Mask should be shape [H, W, N] where there are N instances
	"""

	"""
	Initial, slow method.
	"""
	# start=time.time()
	# for n in range(binary_mask.shape[2]):
	# 	"""
	# 	Padding is necessary, it doesn't create the edge correctly if not.
	# 	"""
	# 	this_mask=binary_mask[:,:,n]
	# 	if binary_mask.dtype=='bool':
	# 		this_mask = np.pad(this_mask,((1,1),(1,1)),constant_values=(False,False))
	# 	else:
	# 		this_mask = np.pad(this_mask,((1,1),(1,1)),constant_values=(0,0))
	# 	contours = skimage.measure.find_contours(this_mask, 0.5, fully_connected='high') #see documentation for 0.5
	# 	# if len(contours)>1:
	# 	# 	print("Weird problem. More than 1 object in channel. Check mask.")
	# 	# 	import pdb;pdb.set_trace()
	# 	# contours=contours[0]
	# 	for i,contour in enumerate(contours):
	# 		contour[contour<0]=0
	# 		contour[contour>binary_mask.shape[0]]=binary_mask.shape[0]
	# 		rr,cc = skimage.draw.polygon_perimeter(contour[:,0],contour[:,1],shape=border_image.shape)
	# 		border_image[rr,cc]=True
	# #this does several morphological operations which are expensive. Alternatively,
	# #you could do a simple convolution operation or using np.where. But this is fine
	# #for now.
	# end = time.time()
	# print("old border method time = ", end-start)
	"""
	This method is anywhere from 4x to 1.1x faster than previous.
	"""
	border_image = np.zeros(shape=binary_mask.shape[0:2],dtype=binary_mask.dtype)
	#start=time.time()
	for n in range(binary_mask.shape[2]):
		this_mask = binary_mask[:,:,n]
		#plan: Erode, invert, multiply.
		inner_mask = erosion(this_mask,square(3))
		inner_mask = inner_mask<0.5 #should be binary anyway, so this is fine.
		border = inner_mask * this_mask
		border_image += border
	#end = time.time()
	#print("new border method time = ", end-start)


	return border_image

def create_borders_N(binary_mask):
	"""
	Given mask, pull out vertices
	Mask should be shape [H, W, N] where there are N instances
	"""

	"""
	Initial, slow method.
	"""
	# start=time.time()
	# for n in range(binary_mask.shape[2]):
	# 	"""
	# 	Padding is necessary, it doesn't create the edge correctly if not.
	# 	"""
	# 	this_mask=binary_mask[:,:,n]
	# 	if binary_mask.dtype=='bool':
	# 		this_mask = np.pad(this_mask,((1,1),(1,1)),constant_values=(False,False))
	# 	else:
	# 		this_mask = np.pad(this_mask,((1,1),(1,1)),constant_values=(0,0))
	# 	contours = skimage.measure.find_contours(this_mask, 0.5, fully_connected='high') #see documentation for 0.5
	# 	# if len(contours)>1:
	# 	# 	print("Weird problem. More than 1 object in channel. Check mask.")
	# 	# 	import pdb;pdb.set_trace()
	# 	# contours=contours[0]
	# 	for i,contour in enumerate(contours):
	# 		contour[contour<0]=0
	# 		contour[contour>binary_mask.shape[0]]=binary_mask.shape[0]
	# 		rr,cc = skimage.draw.polygon_perimeter(contour[:,0],contour[:,1],shape=border_image.shape)
	# 		border_image[rr,cc]=True
	# #this does several morphological operations which are expensive. Alternatively,
	# #you could do a simple convolution operation or using np.where. But this is fine
	# #for now.
	# end = time.time()
	# print("old border method time = ", end-start)
	"""
	This method is anywhere from 4x to 1.1x faster than previous.
	"""
	border_image = np.zeros(shape=binary_mask.shape,dtype=binary_mask.dtype)
	#start=time.time()
	for n in range(binary_mask.shape[2]):
		this_mask = binary_mask[:,:,n]
		#plan: Erode, invert, multiply.
		inner_mask = erosion(this_mask,square(3))
		inner_mask = inner_mask<0.5 #should be binary anyway, so this is fine.
		border_image[:,:,n] = inner_mask * this_mask
		#border_image += border

	#also run _project_masks_rgb(border_image)
	#output = _project_masks_N(border_image, outCH)
	#import pdb;pdb.set_trace()
	return border_image


def load_image_gt(dataset, config, image_id, augmentation=None):
	"""Load and return ground truth data for an image (image, mask, xgrad, ygrad).
	augmentation should be passed as a DICTIONARY, from config.
	Returns:
	image: [height, width, 3] for 2D, [height, width, Z, 1] for 3D
	mask: [height, width]
	xgrad: [height, width]
	ygrad: [height, width]

	"""
	############################################################################
	####################### LOAD IMAGE AND MASK ################################
	############################################################################
	# good_load = False #put in flag for while loop
	# #requires that images have at least one object in them.
	#while not good_load:
	mask = dataset.load_mask(image_id)

	stacked_var = 1

	# Load image and mask
	if hasattr(config,"INPUT_DIM"):
		"""
		Dataset is from CellDataset
		"""
		if config.INPUT_DIM=="3D":
			image = dataset.load_image(image_id, config.INPUT_DIM, mask=mask) #shape is HxWxZ
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
			cents, zlocs, imshape = dataset.load_z_positions(image_id)
			#switch the rows, colums of cents.
			cents = cents[:,::-1]
			#cents is an array of shape N x 2
			#zlocs is a vector array of shape N
			#imshape is a list with length 3.

			#we have to do these operations on the whole image
			#in order for the z-class assignment to be reproducible!
			if augmentation:
				if config.Augmentors['Zflip']:
					if np.random.rand()>=0.5:
						image = image[:,:,::-1,:]
						#flip zlocs.
						zlocs = (imshape[2]-1) - zlocs

				if config.Augmentors['stack']:
					#try to stack
					if int(np.floor(config.INPUT_Z / image.shape[2]))>1:
						stacked_var+=1
						image, mask, zlocs, cents = stack_augmentation(image, mask, zlocs, cents, config.INPUT_Z)

				if config.Augmentors['dense']:
					#could repeat this operation N times to make it dense.
					#we don't want to do this operation if we've stacked the image,
					#might make it too dense.
					if stacked_var==1:
						image, mask, zlocs, cents = dense_translate_augmentation(image, mask, zlocs, cents, config.INPUT_Z, config.OUT_CHANNELS)
			###############################################################
			###############################################################
			#pad the images
			image = dataset.pad_z_image(image, z_to = config.INPUT_Z, center = config.Padding_Opts['center'], random_pad = config.Padding_Opts['random'])
		else: #2D dataset!
			if config.project_mode=="max": #doing max projection!
				image = dataset.load_image(image_id, config.INPUT_DIM, mask=mask)
				cents, zlocs, imshape = dataset.load_z_positions(image_id)
				#switch the rows, colums of cents.
				cents = cents[:,::-1]
			else:
				#mode is "linear". Encode depth.
				image = dataset.load_image(image_id, "3D", mask=mask) #shape is HxWxZx1
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
				cents, zlocs, imshape = dataset.load_z_positions(image_id)
				#switch the rows, colums of cents.
				cents = cents[:,::-1]
				#cents is an array of shape N x 2
				#zlocs is a vector array of shape N
				#imshape is a list with length 3.

				#we have to do these operations on the whole image
				#in order for the z-class assignment to be reproducible!
				if augmentation:
					if config.Augmentors['Zflip']:
						if np.random.rand()>=0.5:
							image = image[:,:,::-1,:]
							#flip zlocs.
							zlocs = (imshape[2]-1) - zlocs

					if config.Augmentors['stack']:
						#try to stack
						if int(np.floor(config.INPUT_Z / image.shape[2]))>1:
							stacked_var+=1
							image, mask, zlocs, cents = stack_augmentation(image, mask, zlocs, cents, config.INPUT_Z)

					if config.Augmentors['dense']:
						#could repeat this operation N times to make it dense.
						#we don't want to do this operation if we've stacked the image,
						#might make it too dense.
						if stacked_var==1:
							image, mask, zlocs, cents = dense_translate_augmentation(image, mask, zlocs, cents, config.INPUT_Z, config.OUT_CHANNELS)
				###############################################################
				###############################################################

				image = linearly_color_encode_image(image)
				#pad the images
				#image = dataset.pad_z_image(image, z_to = config.INPUT_Z, center = config.Padding_Opts['center'], random_pad = config.Padding_Opts['random'])
				#image = linearly_color_encode_image(image)
			#image is 2D
	else:
		"""
		Dataset is from CellPoseDataset
		"""
		image = dataset.load_image(image_id)

	##########################################################################
	######################### APPLY AUGMENTATIONS ############################
	##########################################################################

	# Augmentation
	# This requires the imgaug lib (https://github.com/aleju/imgaug)
	used_list = [] #names of used augmentations #necssary to have outside of if statement
	#because validation does not use augmentation and throws an error.
	if augmentation:
		import imgaug
		import imgaug.augmenters as iaa
		from imgaug import parameters as iap
		#imgaug requires :
		# 'images' should be either a 4D numpy array of shape (N, height, width, channels)
		# or a list of 3D numpy arrays, each having shape (height, width, channels).
		# Grayscale images must have shape (height, width, 1) each.
		# All images must have numpy's dtype uint8. Values are expected to be in
		# range 0-255 integers.
		#(H,W,C) ndarray or (H,W) ndarray

		#clever idea: determine probabilities of selecting particular augmentors BEFORE.
		#this will need to go into: model.py BEFORE MASK_AUGMENTORS line.
		#set up a list of augmentors.
		#see imgaug lambda for adding your own new augmentation, if necessary.
		#select one of these:
		augment_list=[]
		P_use=[]

		if augmentation['XYflip']:
			image, mask, cents, _ = image_xyflip(image, mask, cents, 0.5)
			# if np.random.rand()>=0.5:
			# 	image
			# augment_list.append(iaa.Fliplr(1,name="fliplr"))
			# augment_list.append(iaa.Flipud(1,name="flipud"))
			# P_use.append(0.5)
			# P_use.append(0.5)

		if augmentation['rotate']:
			# # affine_list = [iaa.Affine(rotate=90,name="rot90"),
			# # 			   iaa.Affine(rotate=180,name="rot180"),
			# # 			   iaa.Affine(rotate=270,name="rot270")]
			# affine_list = [iaa.Affine(rotate=90,name="rot90")]
			# N = np.random.rand(1,len(affine_list))[0]
			# N = N/np.sum(N)
			# N = N.tolist()
			# #pick affine
			# affine_pick = affine_list[N.index(max(N))]
			# augment_list.append(affine_pick)
			# P_use.append(0.5)
			image, mask, cents, _ = image_mask_rotate(image, mask, cents, 0.5)

		if augmentation['shear']:
			shear_deg = np.random.uniform(-8,8)
			shear = iaa.Affine(
							shear=shear_deg,
							name="shear")
			augment_list.append(shear)
			P_use.append(0.9)

		if augmentation['zoom']:
			#scalex = np.random.uniform(0.2,1.1) #was 0.2 for 4x
			scalex = np.random.uniform(0.4,0.6)
			#20x is at 0.538 microns/pixel
			#10x is at 1.075 microns/pixel
			zoom = iaa.Affine(
							scale={"x": scalex, "y": scalex},
							name="zoom")
			augment_list.append(zoom)
			P_use.append(1)

		if augmentation['blur']:
			# augment_list.append(iaa.AverageBlur(k=(1,4),name="blur"))
			# P_use.append(0.5) #so 75% of the time, it is normal.
			# augment_list.append(iaa.GaussianBlur(
		    #     sigma=iap.Uniform(0.0, 1.0)))
			# P_use.append(0.5)
			if np.random.rand()>=0.5:
				from scipy.ndimage import gaussian_filter
				n_sigma = np.random.uniform(low=0,high=2)
				image = gaussian_filter(image, sigma=n_sigma)
				#So when doing this, do we want to set the masks to zero?
				used_list.append('blur')

		if augmentation['blur_out']:
			if np.random.rand()>=0.75: #not very probable
				from scipy.ndimage import gaussian_filter
				n_sigma = np.random.uniform(low=2.5,high=6)
				image = gaussian_filter(image, sigma=n_sigma)
				#So when doing this, do we want to set the masks to zero?
				used_list.append('blur_out')

		if augmentation['brightness']:
			# augment_list.append(iaa.Multiply((0.5,1.2),name="multiply"))
			# #this may exceed the expected max of 1 as an input image.
			# #may saturate the image.
			# P_use.append(0.25)
			if np.random.rand()>0.5:
				r_add = np.random.uniform(low=0.1,high=0.5)
				image = image + r_add
				#used_dic['brightness_add']=r_add
				used_list.append('brightness')

		if augmentation['blend']:
			# augment_list.append(iaa.BlendAlphaSimplexNoise(
			#     foreground=iaa.EdgeDetect(1.0),
			#     per_channel=True),
			# 	name="blend")
			# P_use.append(1.0)
			augment_list.append(iaa.BlendAlphaFrequencyNoise(
			    exponent=-3,
			    foreground=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=False),
			    size_px_max=32,
			    upscale_method="linear",
			    iterations=1,
			    sigmoid=False,
				name="blend"))
			P_use.append(0.5)

		if augmentation['noise']:
			augment_list.append(iaa.MultiplyElementwise((0.75, 1.25),name="noise"))
			P_use.append(0.5)

		if augmentation['contrast']:
			# augment_list.append(iaa.GammaContrast((0.5, 2.0),name="contrast"))
			# P_use.append(0.5)

			augment_list.append(iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6),name="sigcontrast"))
			P_use.append(0.5)

			augment_list.append(iaa.LogContrast(gain=(0.6, 1.4),name="logcontrast"))
			P_use.append(0.5)

		#pick N (length of augment_list) random numbers.
		N = np.random.rand(1,len(augment_list))[0]
		#take indices where N > P_use
		inds = [i for i,x in enumerate(N) if x<=P_use[i]]
		augment_used = [x for i,x in enumerate(augment_list) if i in inds]
		used_list = [x.name for x in augment_used] + used_list
		#print(used_list)

		if config.INPUT_DIM=="3D":
			if augmentation['Zflip']:
				if np.random.rand()>0.5:
					used_list.append('Zflip')
		########################################################################
		########################################################################

		#################ASSIGN CELLS TO OUTPUT CHANNELS########################
		if hasattr(config,"INPUT_DIM"):
			zclasses = dataset.load_z_class_alt2(cents, zlocs, image.shape,
						config.OUT_CHANNELS, config.GT_ASSIGN, zsize = config.INPUT_Z,
						zflip = True if 'Zflip' in used_list else False,
						yflip = True if 'fliplr' in used_list else False,
						xflip = True if 'flipud' in used_list else False,
						xyflip = True if 'rot90' in used_list else False)
		else:
			#data is from CellPoseDataset
			zclasses = dataset.load_z_class_alt(image_id, config.OUT_CHANNELS,
			 			config.GT_ASSIGN, zsize = config.INPUT_Z,
						n_stacked = stacked_var, made_dense = dense_var,
						zflip = True if 'Zflip' in used_list else False,
						yflip = True if 'fliplr' in used_list else False,
						xflip = True if 'flipud' in used_list else False,
						xyflip = True if 'rot90' in used_list else False)
		######################################################################

		if 'Zflip' in used_list:
			image = image[:,:,::-1,:]
			#image = np.flip(image,axis=2)

		"""
		Some very important notes: CE: It would seem that if we augment the gradients,
		that the incorrect things would be learned. For example, if we want the first channel to learn
		the x gradient (left to right), if we did a 180 degree flip, you would have + on
		the left and - on the right, which is different from the usual - on the left + on the right.
		THIS HAS BEEN RESOLVED BY CALCULATING GRADIENTS AFTER AUGMENTATIONS, CE
		"""

		aug = iaa.Sequential(augment_used)

		det = aug.to_deterministic()

		MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
						   "Fliplr", "Flipud", "CropAndPad",
						   "Affine", "PiecewiseAffine"]

		#bad_mask_augmenters=["blur","AverageBlur","Multiply","multiply"]
		#import pdb;pdb.set_trace()

		def hook(images, augmenter, parents, default):
			"""Determines which augmenters to apply to masks."""
			return augmenter.__class__.__name__ in MASK_AUGMENTERS

		# Store shapes before augmentation to compare
		image_shape = image.shape
		mask_shape = mask.shape
		import warnings
		warnings.filterwarnings('ignore')#,".*SuspiciousSingleImageShapeWarning.*")
		#roll axis
		"""
		It should be noted that imgaug is tested mostly for uint8 images, but it does
		offer support for float32 except some features may fail at LARGE numbers.
		Since our numbers are limited to small numbers ~1, I have tested the augmentations
		and they seem fine. For more info, see imgaug "dtype_support".
		"""
		image = np.rollaxis(image, 2, 0) #this may flip the image...
		#import pdb;pdb.set_trace()
		image = det.augment_images(image) #NOTE THE augment_images FUNCTION!
		#This is nice because, regardless if it is a z-stack image, with shape
		#Z x H x W x 1, or a 2D image with shape 3 x H x W [after rolling axis above],
		#the operation here will act on the Z images if 3D, and the N channels, if 2D.
		mask = det.augment_image(mask.astype(np.uint8),hooks=imgaug.HooksImages(activator=hook))
		#roll axis back
		image = np.rollaxis(image, 0, 3)

		# #############
		# #crop from center
		# #############
		# if image.shape!=image_shape:
		# 	#in it's current form, this does nothing.
		# 	iy,ix=image.shape[0:2] #image and mask initially should be same shape.
		# 	image = crop_center(image, ix, iy)
		# 	mask = crop_center(mask, ix, iy)

		assert image.shape == image_shape, "Augmentation shouldn't change image size"
		assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
		# Change mask back to bool
		mask = mask>0.5
		#mask = mask.astype(np.bool)

	else:
		#################ASSIGN CELLS TO OUTPUT CHANNELS###################
		if hasattr(config,"INPUT_DIM"):
			zclasses = dataset.load_z_class_alt2(cents, zlocs, image.shape,
						config.OUT_CHANNELS, config.GT_ASSIGN, zsize = config.INPUT_Z)
		else:
			#data is from CellPoseDataset
			zclasses = dataset.load_z_class_alt(image_id, config.OUT_CHANNELS, config.GT_ASSIGN, n_stacked = stacked_var, made_dense=dense_var, zsize = config.INPUT_Z)
		###################################################################

	###########################################################################
	############################### CROP ######################################
	###########################################################################
	""" Now, we will crop the image/mask pair to appropriate shape """
	#Make necessary copies
	image_full = copy.deepcopy(image)
	mask_full = copy.deepcopy(mask)
	zclasses_full = copy.deepcopy(zclasses)

	good_load = False
	#import pdb;pdb.set_trace()
	while not good_load:
		#Do the resizing
		image, window, scale, padding, crop = resize_image(
							image_full, cents, min_dim=config.IMAGE_MIN_DIM, \
							max_dim=config.IMAGE_MAX_DIM, \
							min_scale=config.IMAGE_MIN_SCALE, \
							mode=config.IMAGE_RESIZE_MODE)

		mask = resize_mask(mask_full, scale, padding, crop=crop)
		#now, since it was augmented, sometimes the masks will not be part of the image.
		#Go through the images of mask, delete those that have all zeros
		#filter 1
		_idx1 = np.sum(mask, axis=(0, 1)) > 10
		"""
		What do we do if mask is totally empty, that is a segment was picked that has nothing in it?
		One easy option is just to return zeros for mask, xgrad, and ygrad.
		"""
		if any(np.ravel(_idx1)):
			mask = mask[:, :, _idx1]
			zclasses = zclasses_full[_idx1]
			#what if there are more than two objects in one image?
			#this happens sometimes with random croppings.
			"""
			Need to check each channel for multiple objects.
			This will be computationally expensive.
			Label each image. append if more than one.
			"""
			for obj in range(mask.shape[2]):
				#label image.
				mask_temp,n = skimage.measure.label(mask[:,:,obj],return_num=True)#,connectivity=2)
				if n>1:
					for n in range(2,int(np.max(mask_temp))+1):
						mask_put = mask_temp==n
						#append to mask
						#delete from mask[:,:,c]=
						#import pdb;pdb.set_trace()
						"""
						What can happen here is the big object gets deleted, leaving just a few pixels behind.
						"""
						mask[:,:,obj]=np.abs(mask[:,:,obj].astype(np.float32)-mask_put.astype(np.float32)).astype(np.bool)
						#only do this if the sum is bigger than some threshold.
						#import pdb;pdb.set_trace()
						if np.sum(mask_put)>10:
							mask=np.append(mask,np.expand_dims(mask_put,axis=-1),axis=2)
							zclasses = np.concatenate([zclasses, np.array([zclasses[obj]])]) #this might be incorrect.

			#filter 2, for small leftover parts.
			_idx2 = np.sum(mask, axis=(0, 1)) > 10
			#what if this is ALL zeros?
			#_idx3 = np.sum(mask, axis=(0, 1)) > 400 #require a large object.
			if not any(np.ravel(_idx2)):
				good_load = False
			else:
				good_load = True
				mask = mask[:, :, _idx2]
				zclasses = zclasses[_idx2]

		else:
			good_load = False

		#Let's add one more check. I want to avoid images with mostly background.
		#it would be good if at least 10% of the max proj image was not background.
		#Perhaps a crowding augmentation would be nice.
		#10% is maybe a big ask. A single cell in an image is roughly 2% of the image.
		#Do this in the future. CE 06/14/21

	###########################################################################
	###########################################################################
	###########################################################################

	###ANOTHER Augmentation opportunity.

	# CLIP IMAGE MAX. If brightness or multiply happens, this adjusts it.
	#this can saturate the image.
	#image = np.clip(image, 0., 1.)

	#Normalize instead of clip.
	#image /= np.max(image)

	if np.random.rand()>=0.6:
		#clip the image
		#import pdb;pdb.set_trace()
		image = np.clip(image, 0., 1.)
		used_list.append('image_clip')
	else:
		#normalize the image
		image /= np.max(image)
		image = np.clip(image,0.,1.)
		used_list.append('image_norm')

	#########################################################################
	####################### CREATE GT OUTPUTS ###############################
	#########################################################################
	"""
	Calculate gradients and edges for each object, then place each object into
	the appropriate output channel based on zclasses.
	"""
	mask, borders, xgrad, ygrad = _project_all_by_Z(mask, zclasses, config.OUT_CHANNELS)

	#finally, if we used the blur_out augmentation, then we applied a blur that
	#was so heavy, we shouldn't be able to really see any ground truth elements.
	if 'blur_out' in used_list:
		#make all zeros.
		mask = np.zeros_like(mask)
		borders = np.zeros_like(borders)
		xgrad = np.zeros_like(xgrad)
		ygrad = np.zeros_like(ygrad)

	return image, mask, xgrad*10., ygrad*10., borders
	#return image, mask, xgrad, ygrad, borders

def load_image_gt2(dataset, config, image_id, augmentation=None):
	"""Load and return ground truth data for an image (image, mask, xgrad, ygrad).
	augmentation should be passed as a DICTIONARY, from config.
	Returns:
	image: [height, width, 3] for 2D, [height, width, Z, 1] for 3D
	mask: [height, width]
	xgrad: [height, width]
	ygrad: [height, width]
	Here, I want to do the cropping early, before applying the augmentations.
	(1) this will speed things up.
	(2) By using the centroid cropping method, and applying the 'dense' or 'stack'
	augmentation, I can guarentee that cells will be on top of each other in every image.
	(3) Hopefully, this will make the GT assignment adaptive. In the previous version,
	the gt assignment is based on the overall picture and has less to do with the individual
	small cropped image seen during training. If the training image is instead used,
	hopefully this will result in some better adaptation.
	So to be clear we (1) want to load the image and mask, centroids, zlocs, etc,
	(2)Then do the cropping.
	(3)Then do the augmentations.
	(4)Then do padding!
	(5)Then do the gt assignment/find gradients stuff.

	###########################################################################
	#This method reduces the computation time by 38% compared to load_image_gt#
	###########################################################################

	The problem with this method, if there is one, is that the ground truth assignment
	is not kept constant depending purely on the z-location -- it also depends upon how
	many cells end up in the image.
	Again, hopefully this makes the method more adaptable, but there is no free cake.
	Additionally, if the augmentations of stack or dense are set to True, we end up
	seeing the same cells multiple times in the same image which if a shape is biased -
	that is, we see more if it as is - than that bias will be blown up even worse.
	"""
	############################################################################
	####################### LOAD IMAGE AND MASK ################################
	############################################################################
	mask = dataset.load_mask(image_id)

	stacked_var = 1
	# Load image and mask
	if hasattr(config,"INPUT_DIM"):
		"""
		Dataset is from CellDataset
		"""
		if config.INPUT_DIM=="3D":
			image = dataset.load_image(image_id, config.INPUT_DIM, mask=mask) #shape is HxWxZ
			cents, zlocs, imshape = dataset.load_z_positions(image_id)
			#switch the rows, colums of cents.
			cents = cents[:,::-1]
			#cents is an array of shape N x 2
			#zlocs is a vector array of shape N
			#imshape is a list with length 3.

		else: #2D dataset!
			if config.project_mode=="max": #doing max projection!
				image = dataset.load_image(image_id, config.INPUT_DIM, mask=mask)
				cents, zlocs, imshape = dataset.load_z_positions(image_id)
				#switch the rows, colums of cents.
				cents = cents[:,::-1]
			else:
				#mode is "linear". Encode depth.
				image = dataset.load_image(image_id, "3D", mask=mask) #shape is HxWxZx1
				cents, zlocs, imshape = dataset.load_z_positions(image_id)
				#switch the rows, colums of cents.
				cents = cents[:,::-1]
				#cents is an array of shape N x 2
				#zlocs is a vector array of shape N
				#imshape is a list with length 3.
			#image is 2D
	else:
		"""
		Dataset is from CellPoseDataset
		"""
		image = dataset.load_image(image_id)

	###########################################################################
	############### APPLY ZOOM AUGMENTATION IF THERE IS A ZOOM ################
	###########################################################################
	"""
	We do it at this stage because if done after cropping than there are few cells in the image.
	More computationally expensive to do it here, but oh well.
	"""
	if augmentation:
		if augmentation['zoom']:
			import imgaug
			import imgaug.augmenters as iaa
			from imgaug import parameters as iap
			#scalex = np.random.uniform(0.2,1.1) #was 0.2 for 4x
			scalex = np.random.uniform(0.4,0.6) #was 0.2 for 4x
			#20x is at 0.538 microns/pixel
			#10x is at 1.075 microns/pixel
			zoom = iaa.Affine(
							scale={"x": scalex, "y": scalex},
							name="zoom")
			augment_used = [zoom]
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
			import warnings
			warnings.filterwarnings('ignore')#,".*SuspiciousSingleImageShapeWarning.*")
			#roll axis
			"""
			It should be noted that imgaug is tested mostly for uint8 images, but it does
			offer support for float32 except some features may fail at LARGE numbers.
			Since our numbers are limited to small numbers ~1, I have tested the augmentations
			and they seem fine. For more info, see imgaug "dtype_support".
			"""
			image = np.rollaxis(image, 2, 0) #this may flip the image...
			image = det.augment_images(image) #NOTE THE augment_images FUNCTION!
			#This is nice because, regardless if it is a z-stack image, with shape
			#Z x H x W x 1, or a 2D image with shape 3 x H x W [after rolling axis above],
			#the operation here will act on the Z images if 3D, and the N channels, if 2D.
			mask = det.augment_image(mask.astype(np.uint8),hooks=imgaug.HooksImages(activator=hook))
			#roll axis back
			image = np.rollaxis(image, 0, 3)

			assert image.shape == image_shape, "Augmentation shouldn't change image size"
			assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
			# Change mask back to bool
			mask = mask>0.5

			#readjust centroid positions.
			sx = scalex * image.shape[0]
			sy = scalex * image.shape[1]
			xmin = image.shape[0]//2 - sx//2
			ymin = image.shape[1]//2 - sy//2
			cents = cents * scalex
			cents[:,1] = cents[:,1] + xmin
			cents[:,0] = cents[:,0] + ymin

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
		image, window, scale, padding, crop = resize_image(
							image_full, cents, min_dim=config.IMAGE_MIN_DIM, \
							max_dim=config.IMAGE_MAX_DIM, \
							min_scale=config.IMAGE_MIN_SCALE, \
							mode=config.IMAGE_RESIZE_MODE)

		mask = resize_mask(mask_full, scale, padding, crop=crop)
		#now, since it was augmented, sometimes the masks will not be part of the image.
		#Go through the images of mask, delete those that have all zeros
		#filter 1
		_idx1 = np.sum(mask, axis=(0, 1)) > 10
		"""
		What do we do if mask is totally empty, that is a segment was picked that has nothing in it?
		One easy option is just to return zeros for mask, xgrad, and ygrad.
		"""
		if any(np.ravel(_idx1)):
			mask = mask[:, :, _idx1]
			zlocs = zlocs_full[_idx1]
			cents = cents_full[_idx1,:]
			#what if there are more than two objects in one image?
			#this happens sometimes with random croppings.
			"""
			Need to check each channel for multiple objects.
			This will be computationally expensive.
			Label each image. append if more than one.
			"""
			for obj in range(mask.shape[2]):
				#label image.
				mask_temp,n = skimage.measure.label(mask[:,:,obj],return_num=True)#,connectivity=2)
				if n>1:
					for n in range(2,int(np.max(mask_temp))+1):
						mask_put = mask_temp==n
						#append to mask
						"""
						What can happen here is the big object gets deleted, leaving just a few pixels behind.
						"""
						mask[:,:,obj]=np.abs(mask[:,:,obj].astype(np.float32)-mask_put.astype(np.float32)).astype(np.bool)
						#only do this if the sum is bigger than some threshold.
						if np.sum(mask_put)>10:
							mask=np.append(mask,np.expand_dims(mask_put,axis=-1),axis=2)
							zlocs = np.concatenate([zlocs, np.array([zlocs[obj]])]) #this might be incorrect.
							cents = np.concatenate([cents, np.expand_dims(cents[obj,:],axis=0)])
			#filter 2, for small leftover parts.
			_idx2 = np.sum(mask, axis=(0, 1)) > 10
			#what if this is ALL zeros?
			#_idx3 = np.sum(mask, axis=(0, 1)) > 400 #require a large object.
			if not any(np.ravel(_idx2)):
				good_load = False
			else:
				good_load = True
				mask = mask[:, :, _idx2]
				zlocs = zlocs[_idx2]
				cents = cents[_idx2,:]

		else:
			good_load = False

	###########################################################################
	###########################################################################
	###########################################################################

	##########################################################################
	######################### APPLY AUGMENTATIONS ############################
	##########################################################################

	# Augmentation
	# This requires the imgaug lib (https://github.com/aleju/imgaug)
	used_list = [] #names of used augmentations #necssary to have outside of if statement
	#because validation does not use augmentation and throws an error.
	if augmentation:
		import imgaug
		import imgaug.augmenters as iaa
		from imgaug import parameters as iap
		#imgaug requires :
		# 'images' should be either a 4D numpy array of shape (N, height, width, channels)
		# or a list of 3D numpy arrays, each having shape (height, width, channels).
		# Grayscale images must have shape (height, width, 1) each.
		# All images must have numpy's dtype uint8. Values are expected to be in
		# range 0-255 integers.
		#(H,W,C) ndarray or (H,W) ndarray

		#clever idea: determine probabilities of selecting particular augmentors BEFORE.
		#this will need to go into: model.py BEFORE MASK_AUGMENTORS line.
		#set up a list of augmentors.
		#see imgaug lambda for adding your own new augmentation, if necessary.
		#select one of these:

		if np.logical_or(config.INPUT_DIM=="3D", np.logical_and(config.INPUT_DIM=="2D", config.project_mode=="linear" if hasattr(config,'project_mode') else False)):
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
			if config.Augmentors['Zflip']:
				if np.random.rand()>=0.5:
					image = image[:,:,::-1,:]
					#flip zlocs.
					zlocs = (imshape[2]-1) - zlocs

			if config.Augmentors['stack']:
				#try to stack
				if int(np.floor(config.INPUT_Z / image.shape[2]))>1:
					stacked_var+=1
					image, mask, zlocs, cents = stack_augmentation(image, mask, zlocs, cents, config.INPUT_Z)

			if config.Augmentors['dense']:
				#could repeat this operation N times to make it dense.
				#we don't want to do this operation if we've stacked the image,
				#might make it too dense.
				if stacked_var==1:
					image, mask, zlocs, cents = dense_translate_augmentation(image, mask, zlocs, cents, config.INPUT_Z, config.OUT_CHANNELS)
			###############################################################
			###############################################################
		augment_list=[]
		P_use=[]

		if augmentation['XYflip']:
			image, mask, cents, _ = image_xyflip(image, mask, cents, 0.5)

		if augmentation['rotate']:
			image, mask, cents, _ = image_mask_rotate(image, mask, cents, 0.5)

		if augmentation['shear']:
			shear_deg = np.random.uniform(-8,8)
			shear = iaa.Affine(
							shear=shear_deg,
							name="shear")
			augment_list.append(shear)
			P_use.append(0.9)

		if augmentation['blur']:
			if np.random.rand()>=0.5:
				from scipy.ndimage import gaussian_filter
				n_sigma = np.random.uniform(low=0,high=2)
				image = gaussian_filter(image, sigma=n_sigma)
				#So when doing this, do we want to set the masks to zero?
				used_list.append('blur')

		if augmentation['blur_out']:
			if np.random.rand()>=0.75: #not very probable
				from scipy.ndimage import gaussian_filter
				n_sigma = np.random.uniform(low=2.5,high=6)
				image = gaussian_filter(image, sigma=n_sigma)
				#So when doing this, do we want to set the masks to zero?
				used_list.append('blur_out')

		if augmentation['brightness']:
			if np.random.rand()>0.5:
				r_add = np.random.uniform(low=0.1,high=0.5)
				image = image + r_add
				used_list.append('brightness')

		if augmentation['blend']:
			augment_list.append(iaa.BlendAlphaFrequencyNoise(
			    exponent=-3,
			    foreground=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=False),
			    size_px_max=32,
			    upscale_method="linear",
			    iterations=1,
			    sigmoid=False,
				name="blend"))
			P_use.append(0.5)

		if augmentation['noise']:
			augment_list.append(iaa.MultiplyElementwise((0.75, 1.25),name="noise"))
			P_use.append(0.5)

		if augmentation['contrast']:
			augment_list.append(iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6),name="sigcontrast"))
			P_use.append(0.5)

			augment_list.append(iaa.LogContrast(gain=(0.6, 1.4),name="logcontrast"))
			P_use.append(0.5)

		#pick N (length of augment_list) random numbers.
		N = np.random.rand(1,len(augment_list))[0]
		#take indices where N > P_use
		inds = [i for i,x in enumerate(N) if x<=P_use[i]]
		augment_used = [x for i,x in enumerate(augment_list) if i in inds]
		used_list = [x.name for x in augment_used] + used_list
		#print(used_list)

		if config.INPUT_DIM=="3D":
			if augmentation['Zflip']:
				if np.random.rand()>0.5:
					used_list.append('Zflip')
		########################################################################
		########################################################################

		#################ASSIGN CELLS TO OUTPUT CHANNELS########################
		if hasattr(config,"INPUT_DIM"):
			zclasses = dataset.load_z_class_alt2(cents, zlocs, image.shape,
						config.OUT_CHANNELS, config.GT_ASSIGN, zsize = config.INPUT_Z,
						zflip = True if 'Zflip' in used_list else False,
						yflip = True if 'fliplr' in used_list else False,
						xflip = True if 'flipud' in used_list else False,
						xyflip = True if 'rot90' in used_list else False)
		else:
			#data is from CellPoseDataset
			zclasses = dataset.load_z_class_alt(image_id, config.OUT_CHANNELS,
			 			config.GT_ASSIGN, zsize = config.INPUT_Z,
						n_stacked = stacked_var, made_dense = dense_var,
						zflip = True if 'Zflip' in used_list else False,
						yflip = True if 'fliplr' in used_list else False,
						xflip = True if 'flipud' in used_list else False,
						xyflip = True if 'rot90' in used_list else False)
		######################################################################

		if 'Zflip' in used_list:
			image = image[:,:,::-1,:]

		"""
		Some very important notes: CE: It would seem that if we augment the gradients,
		that the incorrect things would be learned. For example, if we want the first channel to learn
		the x gradient (left to right), if we did a 180 degree flip, you would have + on
		the left and - on the right, which is different from the usual - on the left + on the right.
		THIS HAS BEEN RESOLVED BY CALCULATING GRADIENTS AFTER AUGMENTATIONS, CE
		"""

		aug = iaa.Sequential(augment_used)

		det = aug.to_deterministic()

		MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
						   "Fliplr", "Flipud", "CropAndPad",
						   "Affine", "PiecewiseAffine"]

		#bad_mask_augmenters=["blur","AverageBlur","Multiply","multiply"]
		#import pdb;pdb.set_trace()

		def hook(images, augmenter, parents, default):
			"""Determines which augmenters to apply to masks."""
			return augmenter.__class__.__name__ in MASK_AUGMENTERS

		# Store shapes before augmentation to compare
		image_shape = image.shape
		mask_shape = mask.shape
		import warnings
		warnings.filterwarnings('ignore')#,".*SuspiciousSingleImageShapeWarning.*")
		#roll axis
		"""
		It should be noted that imgaug is tested mostly for uint8 images, but it does
		offer support for float32 except some features may fail at LARGE numbers.
		Since our numbers are limited to small numbers ~1, I have tested the augmentations
		and they seem fine. For more info, see imgaug "dtype_support".
		"""
		image = np.rollaxis(image, 2, 0) #this may flip the image...
		#import pdb;pdb.set_trace()
		image = det.augment_images(image) #NOTE THE augment_images FUNCTION!
		#This is nice because, regardless if it is a z-stack image, with shape
		#Z x H x W x 1, or a 2D image with shape 3 x H x W [after rolling axis above],
		#the operation here will act on the Z images if 3D, and the N channels, if 2D.
		mask = det.augment_image(mask.astype(np.uint8),hooks=imgaug.HooksImages(activator=hook))
		#roll axis back
		image = np.rollaxis(image, 0, 3)

		assert image.shape == image_shape, "Augmentation shouldn't change image size"
		assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
		# Change mask back to bool
		mask = mask>0.5

	else:
		#################ASSIGN CELLS TO OUTPUT CHANNELS###################
		if hasattr(config,"INPUT_DIM"):
			zclasses = dataset.load_z_class_alt2(cents, zlocs, image.shape,
						config.OUT_CHANNELS, config.GT_ASSIGN, zsize = config.INPUT_Z)
		else:
			#data is from CellPoseDataset
			zclasses = dataset.load_z_class_alt(image_id, config.OUT_CHANNELS, config.GT_ASSIGN, n_stacked = stacked_var, made_dense=dense_var, zsize = config.INPUT_Z)
		###################################################################

	###ANOTHER Augmentation opportunity.

	# CLIP IMAGE MAX. If brightness or multiply happens, this adjusts it.
	#this can saturate the image.
	#image = np.clip(image, 0., 1.)

	#Normalize instead of clip.
	#image /= np.max(image)

	if np.random.rand()>=0.6:
		#clip the image
		#import pdb;pdb.set_trace()
		image = np.clip(image, 0., 1.)
		used_list.append('image_clip')
	else:
		#normalize the image
		image /= np.max(image)
		image = np.clip(image,0.,1.)
		used_list.append('image_norm')

	#########################################################################
	####################### CREATE GT OUTPUTS ###############################
	#########################################################################
	"""
	Calculate gradients and edges for each object, then place each object into
	the appropriate output channel based on zclasses.
	"""
	mask, borders, xgrad, ygrad = _project_all_by_Z(mask, zclasses, config.OUT_CHANNELS)

	#finally, if we used the blur_out augmentation, then we applied a blur that
	#was so heavy, we shouldn't be able to really see any ground truth elements.
	if 'blur_out' in used_list:
		#make all zeros.
		mask = np.zeros_like(mask)
		borders = np.zeros_like(borders)
		xgrad = np.zeros_like(xgrad)
		ygrad = np.zeros_like(ygrad)


	#########################################################################
	##################### IF 2D and Linear Mode #############################
	#########################################################################
	if np.logical_and(config.INPUT_DIM=="2D", config.project_mode=="linear" if hasattr(config, 'project_mode') else False):
		if config.pad_before_linear:
			#pad first. See config settings.
			image = dataset.pad_z_image(image, z_to = config.INPUT_Z, center = config.Padding_Opts['center'], random_pad = config.Padding_Opts['random'])
		image = linearly_color_encode_image(image)

	#########################################################################
	######################### PAD THE IMAGE #################################
	#########################################################################
	#pad the images
	if config.INPUT_DIM=="3D":
		image = dataset.pad_z_image(image, z_to = config.INPUT_Z, center = config.Padding_Opts['center'], random_pad = config.Padding_Opts['random'])

	return image, mask, xgrad*10., ygrad*10., borders
	#return image, mask, xgrad, ygrad, borders


###########GENERATOR
def data_generator(dataset, config, shuffle=True, augmentation=False,
				   batch_size=1):
	"""A generator that returns images and corresponding target class ids,
	bounding box deltas, and masks.
	dataset: The Dataset object to pick data from
	config: The model config object
	shuffle: If True, shuffles the samples before every epoch
	augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
		For example, passing imgaug.augmenters.Fliplr(0.5) flips images
		right/left 50% of the time.
	batch_size: How many images to return in each call
	Returns a Python generator. Upon calling next() on it, the
	generator returns two lists, inputs and outputs. The contents
	of the lists differs depending on the received arguments:
	inputs list:
	- images: [batch, H, W, 3]
	- gt_masks: [batch, height, width].
	- gt_xgrads: [batch, height, width]
	- gt_ygrads: [batch, height, width]
	outputs list: Usually empty in regular training. But if detection_targets
		is True then the outputs list contains target class_ids, bbox deltas,
		and masks.
	"""
	b = 0  # batch item index
	image_index = -1
	image_ids = np.copy(dataset.image_ids)
	error_count = 0

	# Keras requires a generator to run indefinitely.
	while True:
		try:
			# Increment index to pick next image. Shuffle if at the start of an epoch.
			#print(image_index)
			image_index = (image_index + 1) % len(image_ids)
			if shuffle and image_index == 0:
				np.random.shuffle(image_ids)

			image_id = image_ids[image_index]

			# If the image source is not to be augmented pass None as augmentation
			if not augmentation:
				#dataset.image_info[image_id]['source'] in no_augmentation_sources:
				image, mask, xgrad, ygrad, border = \
				load_image_gt2(dataset, config, image_id, augmentation=False)
			else:
				image, mask, xgrad, ygrad, border = \
				load_image_gt2(dataset, config, image_id, augmentation=config.Augmentors)
				# if config.Augmentors['brightness']:
				# 	#may have adjusted max of image to be to exceed a value of 1.
				# 	#saturate the image instead.
				# 	#these will be confusing cases for the machine.
				# 	image = np.where(image>1.0, 1.0, image)

			# Init batch arrays
			if b == 0:
				batch_images = np.zeros(
					(batch_size,) + image.shape, dtype=np.float32)
				batch_gt_masks = np.zeros(
					(batch_size,) + mask.shape, dtype=mask.dtype)
				batch_gt_xgrads = np.zeros(
					(batch_size,) + xgrad.shape, dtype=xgrad.dtype)
				batch_gt_ygrads = np.zeros(
					(batch_size,) + ygrad.shape, dtype=ygrad.dtype)
				batch_gt_borders = np.zeros(
					(batch_size,) + border.shape, dtype=border.dtype)

			# Add to batch
			batch_images[b] = image #mold_image(image.astype(np.float32), config)
			batch_gt_masks[b] = mask
			batch_gt_xgrads[b] = xgrad
			batch_gt_ygrads[b] = ygrad
			batch_gt_borders[b] = border

			b += 1

			# Batch full?
			if b >= batch_size:
				inputs = [batch_images, batch_gt_masks, batch_gt_xgrads, batch_gt_ygrads, batch_gt_borders]
				outputs = []

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

################################################################################
################################################################################
################################################################################
################################################################################


def resize_image(image, centroids, min_dim=None, max_dim=None, min_scale=None, mode="square"):
	"""Resizes an image keeping the aspect ratio unchanged.
	min_dim: if provided, resizes the image such that it's smaller
		dimension == min_dim
	max_dim: if provided, ensures that the image longest side doesn't
		exceed this value.
	min_scale: if provided, ensure that the image is scaled up by at least
		this percent even if min_dim doesn't require it.
		none: No resizing. Return the image unchanged.
		square: Resize and pad with zeros to get a square image
			of size [max_dim, max_dim].
		pad64: Pads width and height with zeros to make them multiples of 64.
			   If min_dim or min_scale are provided, it scales the image up
			   before padding. max_dim is ignored in this mode.
			   The multiple of 64 is needed to ensure smooth scaling of feature
			   maps up and down the 6 levels of the FPN pyramid (2**6=64).
		crop: Picks random crops from the image. First, scales the image based
			  on min_dim and min_scale, then picks a random crop of
			  size min_dim x min_dim. Can be used in training only.
			  max_dim is not used in this mode.
		center: Picks the crop from the center of the image only.
	Returns:
	image: the resized image
	window: (y1, x1, y2, x2). If max_dim is provided, padding might
		be inserted in the returned image. If so, this window is the
		coordinates of the image part of the full image (excluding
		the padding). The x2, y2 pixels are not included.
	scale: The scale factor used to resize the image
	padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
	"""
	# Keep track of image dtype and return results in the same dtype
	image_dtype = image.dtype
	# Default window (y1, x1, y2, x2) and default scale == 1.
	h, w = image.shape[:2]
	window = (0, 0, h, w)
	scale = 1
	padding = [(0, 0), (0, 0), (0, 0)]
	crop = None
	#print(image.shape)

	if mode == "none":
		return image, window, scale, padding, crop

	# Scale?
	if min_dim:
		# Scale up but not down
		scale = max(1, min_dim / min(h, w))
	if min_scale and scale < min_scale:
		scale = min_scale

	# Does it exceed max dim?
	if max_dim and mode == "square":
		image_max = max(h, w)
		if round(image_max * scale) > max_dim:
			scale = max_dim / image_max

	# Resize image using bilinear interpolation
	if scale != 1:
		#calculate dims appropriately here.
		image = resize(image, (round(h * scale), round(w * scale), *image.shape[2:]),
					   preserve_range=True)

	# Need padding or cropping?
	if mode == "square":
		# Get new height and width
		h, w = image.shape[:2]
		top_pad = (max_dim - h) // 2
		bottom_pad = max_dim - h - top_pad
		left_pad = (max_dim - w) // 2
		right_pad = max_dim - w - left_pad
		padding = [(top_pad, bottom_pad), (left_pad, right_pad)] + [(0,0) for i in range(image.ndim - 2)]
		#padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
		image = np.pad(image, padding, mode='constant', constant_values=0)
		window = (top_pad, left_pad, h + top_pad, w + left_pad)

	elif mode == "pad64":
		h, w = image.shape[:2]
		# Both sides must be divisible by 64
		#assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
		# Height
		if h % 64 > 0:
			max_h = h - (h % 64) + 64
			top_pad = (max_h - h) // 2
			bottom_pad = max_h - h - top_pad
		else:
			top_pad = bottom_pad = 0
		# Width
		if w % 64 > 0:
			max_w = w - (w % 64) + 64
			left_pad = (max_w - w) // 2
			right_pad = max_w - w - left_pad
		else:
			left_pad = right_pad = 0
		padding = [(top_pad, bottom_pad), (left_pad, right_pad)] + [(0,0) for i in range(image.ndim - 2)]
		#[(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
		image = np.pad(image, padding, mode='constant', constant_values=0)
		window = (top_pad, left_pad, h + top_pad, w + left_pad)

	elif mode == "crop":
		# Pick a random crop
		h, w = image.shape[:2]
		#good_pick=False
		#while not good_pick:
		y = random.randint(0, (h - min_dim))
		x = random.randint(0, (w - min_dim))
		crop = (y, x, min_dim, min_dim)
		#idx = np.sum(mask[y:y + min_dim, x:x + min_dim], axis=(0, 1)) > 0.02*(min_dim*min_dim)
		#if np.sum(idx)>1: #require more than one object in the image.
		#	good_pick=True
		#import pdb;pdb.set_trace()
		image = image[y:y + min_dim, x:x + min_dim]
		window = (0, 0, min_dim, min_dim)

	#added CE 11/24/21
	elif mode == "center":
		h, w = image.shape[:2]
		x = int(w/2)-int(np.floor(min_dim/2))
		y = int(h/2)-int(np.floor(min_dim/2))
		crop = (y, x, min_dim, min_dim)
		image = image[y:y + min_dim, x:x + min_dim]
		window = (0, 0, min_dim, min_dim)

	#added CE 04/28/22
	elif mode== "centroid":
		#because random selects a random portion and we also require that there is
		#a cell in the cropped portion, we end up needing to repeat the operation
		#sometimes. This increases the computation time.
		# pick a cell centroid as the center.
		h, w = image.shape[:2]
		picked = random.randint(0, centroids.shape[0]-1)
		x = int(np.round(centroids[picked,1]))
		y = int(np.round(centroids[picked,0]))
		#lets determine if those positions work, and adjust if not
		if y + np.ceil(min_dim/2) > h:
			y = h - min_dim
		elif y - np.floor(min_dim/2) < 0:
			y = 0
		else:
			y = y - int(np.floor(min_dim/2))

		if x + np.ceil(min_dim/2) > w:
			x = w - min_dim
		elif x - np.floor(min_dim/2) < 0:
			x = 0
		else:
			x = x - int(np.floor(min_dim/2))
		#import pdb;pdb.set_trace()
		crop = (y, x, min_dim, min_dim)
		image = image[y:y + min_dim, x:x + min_dim]
		window = (0, 0, min_dim, min_dim)

	else:
		raise Exception("Mode {} not supported".format(mode))
	#print(image.shape)
	return image, window, scale, padding, crop


def resize_mask(mask, scale, padding, crop=None):
	"""Resizes a mask using the given scale and padding.
	Typically, you get the scale and padding from resize_image() to
	ensure both, the image and the mask, are resized consistently.
	scale: mask scaling factor
	padding: Padding to add to the mask in the form
			[(top, bottom), (left, right), (0, 0)]
	"""
	# Suppress warning from scipy 0.13.0, the output shape of zoom() is
	# calculated with round() instead of int()
	if scale!=1:
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			if len(mask.shape)==3:
				mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
			else:
				mask = scipy.ndimage.zoom(mask, zoom=[scale, scale], order=0)
	if crop is not None:
		y, x, h, w = crop
		mask = mask[y:y + h, x:x + w]
	else:
		mask = np.pad(mask, padding, mode='constant', constant_values=0)
	return mask

def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
		   preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
	"""A wrapper for Scikit-Image resize().
	Scikit-Image generates warnings on every call to resize() if it doesn't
	receive the right parameters. The right parameters depend on the version
	of skimage. This solves the problem by using different parameters per
	version. And it provides a central place to control resizing defaults.
	"""
	if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
		# New in 0.14: anti_aliasing. Default it to False for backward
		# compatibility with skimage 0.13.
		return skimage.transform.resize(
			image, output_shape,
			order=order, mode=mode, cval=cval, clip=clip,
			preserve_range=preserve_range, anti_aliasing=anti_aliasing,
			anti_aliasing_sigma=anti_aliasing_sigma)
	else:
		return skimage.transform.resize(
			image, output_shape,
			order=order, mode=mode, cval=cval, clip=clip,
			preserve_range=preserve_range)


################################################################################
################################################################################
################################################################################

"""
Example Code:
#####3D->2D CODE TEST#####
import config
C = config.Cell2DConfig()
cell_test = CellDataset()
cell_test.load_cell("/users/czeddy/documents/workingfolder/deep_learning_test/cellpose/datasets/mine/train")
cell_test.prepare()
IM, M, X, Y = load_image_gt(cell_test, C, 0, augmentation=True)
fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2)
ax1.imshow(IM)
ax2.imshow(M)
ax3.imshow(X)
ax4.imshow(Y)
plt.show()

####3D CODE TEST#####
import config
C = config.Cell3DConfig()
cell_test = CellDataset()
cell_test.load_cell("/users/czeddy/documents/workingfolder/deep_learning_test/cellpose/datasets/mine/train")
cell_test.prepare()
IM, M, X, Y = load_image_gt(cell_test, C, 0, augmentation=True)
fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2)
ax1.imshow(np.max(IM,axis=2))
ax2.imshow(M)
ax3.imshow(X)
ax4.imshow(Y)
plt.show()

N=np.random.choice(len(cell_test.image_info),1000)
for i,n in enumerate(N):
	print(i)
	IM, M, X, Y = load_image_gt(cell_test, C, n, augmentation=True)
	#check if any of these contain nans
	if np.isnan(np.sum(IM)) or np.isinf(np.sum(IM)):
		print("IM problem")
		import pdb;pdb.set_trace()
	if np.isnan(np.sum(M)) or np.isinf(np.sum(M)):
		print("M problem")
		import pdb;pdb.set_trace()
	if np.isnan(np.sum(X)) or np.isinf(np.sum(X)):
		print("X problem")
		import pdb;pdb.set_trace()
	if np.isnan(np.sum(Y)) or np.isinf(np.sum(Y)):
		print("Y problem")
		import pdb;pdb.set_trace()

####CELL POSE DATASET

import config
C = config.Config()
cell_pose = CellPoseDataset()
cell_pose.load_cell("/users/czeddy/documents/workingfolder/deep_learning_test/cellpose/datasets/MouseVision")
cell_pose.prepare()
IM, M, X, Y = load_image_gt(cell_pose, C, 19, augmentation=True)
fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2)
ax1.imshow(IM)
ax2.imshow(M)
ax3.imshow(X)
ax4.imshow(Y)
plt.show()

N=np.random.choice(len(cell_pose.image_info),1000)
for i,n in enumerate(N):
	print(i)
	IM, M, X, Y = load_image_gt(cell_pose, C, n, augmentation=True)
	#check if any of these contain nans
	if np.isnan(np.sum(IM)):
		print("IM problem")
		import pdb;pdb.set_trace()
	if np.isnan(np.sum(M)):
		print("M problem")
		import pdb;pdb.set_trace()
	if np.isnan(np.sum(X)):
		print("X problem")
		import pdb;pdb.set_trace()
	if np.isnan(np.sum(Y)):
		print("Y problem")
		import pdb;pdb.set_trace()

N = np.random.choice(len(cell_pose.image_info),20)
#make generator
val_generator = data_generator(cell_pose, C, shuffle=True,batch_size=C.BATCH_SIZE)
all_together=[]
for i,n in enumerate(N):
	#make val generator
	print(i)
	O,_=next(val_generator)
	IM, M, X, Y = O
	Msum=np.zeros(C.BATCH_SIZE)
	Xsum=np.zeros(C.BATCH_SIZE)
	Ysum=np.zeros(C.BATCH_SIZE)
	for j in range(M.shape[0]):
		Msum[j]=np.sum(M[j])
		Xsum[j]=np.sum(X[j])
		Ysum[j]=np.sum(Y[j])
	all_together.append([Msum,Xsum,Ysum])
	if np.isnan(np.sum(IM)) or np.isinf(np.sum(IM)):
		print("IM problem")
		import pdb;pdb.set_trace()
	if np.isnan(np.sum(M)) or np.isinf(np.sum(M)):
		print("M problem")
		import pdb;pdb.set_trace()
	if np.isnan(np.sum(X)) or np.isinf(np.sum(X)):
		print("X problem")
		import pdb;pdb.set_trace()
	if np.isnan(np.sum(Y)) or np.isinf(np.sum(Y)):
		print("Y problem")
		import pdb;pdb.set_trace()
Ms_together=np.concatenate([x[0] for x in all_together],axis=0)
Xs_together=np.concatenate([x[1] for x in all_together],axis=0)
Ys_together=np.concatenate([x[2] for x in all_together],axis=0)


"""

def moving_average(x,w):
	"""
	Moving average for background subtract.
	"""
	divisor = np.ones(x.shape)*(w*w)
	#make edges all 6 instead of 9.
	divisor[0,:]=6.
	divisor[-1,:]=6.
	divisor[:,0]=6.
	divisor[:,-1]=6.
	#make corners 4.
	divisor[0,0]=4.
	divisor[-1,0]=4.
	divisor[0,-1]=4.
	divisor[-1,-1]=4.
	return convolve(x, np.ones((w,w)), mode='constant') / divisor

def load_grad(masks, channels):
	"""
	Load mask
	Run code on mask to find centers
	Run code to run heat diffusion
	Run code to calculate x gradient
	Run code to calculate y gradient
	Masks should be HxWxN
	"""
	centers,niters,bboxes = _find_centers(masks)
	xgrads,ygrads = _run_gradient_find(centers,niters,bboxes,masks)
	#xgrad = _max_project_grads(xgrads)
	#xgrad = _max_naive_project_grads(xgrads)
	xgrad = _project_grads_N(xgrads, masks, channels)
	xgrad = xgrad.astype(np.float32)
	#ygrad = _max_project_grads(ygrads)
	ygrad = _project_grads_N(ygrads, masks, channels)
	ygrad = ygrad.astype(np.float32)
	#heat diffusion from center to mask.
	#we only want to really do this once, so we will eventually want to move this outside of the dataset class.

	return xgrad, ygrad

def _max_project_grads(grads):
	"""
	Use to return a single H x W prediction of a grads
	This method takes the largest absolute value of the gradient. This can lead
	to learning flows that will result in two overlapped objects always being just
	one object. Consider two overlapping circles and their respective gradients.
	"""
	A=np.max(grads,axis=2)
	B=np.min(grads,axis=2)
	C = np.where(np.abs(A)>np.abs(B),A,B)
	return C

def _max_val_project_grads(grads):
	"""
	This method uses the maximum projection of gradients...
	See train example 19. This doesn't work well either.
	"""
	#add a slice to grads of all zeros.
	grads = np.insert(grads,0,0.0,axis=2)
	idx = np.argmax(grads,axis=2)
	idx_neg = np.argmin(grads,axis=2)
	idx = np.where(idx==0, idx_neg, idx)
	m,n = grads.shape[:2]
	I,J = np.ogrid[:m,:n]
	output = grads[I,J,idx]
	return output

def _max_naive_project_grads(grads):
	"""
	Simply lay them on top of another another
	"""
	m,n,o = grads.shape
	output = np.zeros(shape=(m,n), dtype=grads.dtype)
	for obj in range(o):
		output = np.where(grads[:,:,obj]!=0, grads[:,:,obj], output)
	#import pdb;pdb.set_trace()
	return output

def _project_grads_N(grads, masks, outCH):
	"""
	V1.1
	Put cell objects in RGB images, force to lowest channel unless there is
	pixel overlap.
	grads is shape H x W x N where N is the number of detections
	Preferentially places objects in the first channel, then second, and so on,
	to prevent overlap.
	"""
	_,_,Nmasks = masks.shape
	H, W, N = grads.shape
	if Nmasks!=N:
		print("Pausing in _project_grads_N...")
		import pdb;pdb.set_trace()
	#outCH = 4 #output channels
	output = np.zeros(shape=(H,W,outCH), dtype=grads.dtype)
	for obj in range(N):
		ch = 0
		placed = False
		while not placed and ch < outCH:
			P = np.argwhere(grads[:,:,obj]!=0.)
			if np.sum([output[r,c,ch] for r,c in P])==0.0:
				#output[:,:,ch]=np.where(grads[:,:,obj]!=0.,grads[:,:,obj],output[:,:,ch])
				output[:,:,ch]=np.where(masks[:,:,obj]==True,grads[:,:,obj],output[:,:,ch])
				placed = True
			else:
				ch += 1
		if ch == outCH and not placed:
			#issue with needing more channels than just 3!
			print("Need more than {} channels for output... Pausing for debugger.".format(outCH))
			import pdb;pdb.set_trace()
	#WORKS, CE 122121
	return output

def _project_masks_N(masks, outCH):
	"""
	V1.4
	masks shape is H x W x N
	should be dtype boolean
	Preferentially places objects in the first channel, then second, and so on.
	"""
	H, W, N = masks.shape
	#outCH = 3
	output = np.zeros(shape=(H,W,outCH), dtype=masks.dtype)
	for obj in range(N):
		ch = 0
		placed = False
		while not placed and ch < outCH:
			P = np.argwhere(masks[:,:,obj]==True)
			if np.sum([output[r,c,ch] for r,c in P])==0.0:
				output[:,:,ch]=np.where(masks[:,:,obj]==True, masks[:,:,obj], output[:,:,ch])
				placed = True
			else:
				ch += 1
		if ch == outCH and not placed:
			#issue with needing more channels than just 3!
			print("Need more than {} channels for output... Pausing for debugger.".format(outCH))
			import pdb;pdb.set_trace()
	#WORKS, CE 122121
	return output

def _project_masks_borders_grads_N(masks, outCH):
	"""
	V1.5
	Randomly places cell objects in the same randomly chosen channel.
	Should resolve issue with preferentially selecting objects to the first channel only.
	Probably requires different loss functions, since prediction should likely
	appear randomly in any channel.
	"""
	H, W, N = masks.shape
	#load borders
	borders = create_borders_N(masks)
	_,_,Nborders = borders.shape
	if N != Nborders:
		print("pausing different number of objects...")
		import pdb;pdb.set_trace()

	outMasks = np.zeros(shape=(H,W,outCH), dtype=masks.dtype)
	outBorders = np.zeros(shape=(H,W,outCH), dtype=borders.dtype)
	channel_list = list(range(outCH))
	#load gradients
	centers,niters,bboxes = _find_centers(masks)
	xgrads,ygrads = _run_gradient_find(centers,niters,bboxes,masks)

	outXGrad = np.zeros(shape=(H,W,outCH), dtype=xgrads.dtype)
	outYGrad = np.zeros(shape=(H,W,outCH), dtype=ygrads.dtype)
	#place objects in ground truth outputs randomly, but they should be in the same channel.
	for obj in range(N):
		ch=0
		random.shuffle(channel_list)
		placed = False
		while not placed and ch < outCH:
			P = np.argwhere(masks[:,:,obj]==True)
			if np.sum([outMasks[r,c,channel_list[ch]] for r,c in P])==0.0:
				outMasks[:,:,channel_list[ch]]=np.where(masks[:,:,obj]==True, masks[:,:,obj], outMasks[:,:,channel_list[ch]])
				outBorders[:,:,channel_list[ch]]=np.where(masks[:,:,obj]==True, borders[:,:,obj], outBorders[:,:,channel_list[ch]])
				outXGrad[:,:,channel_list[ch]]=np.where(masks[:,:,obj]==True, xgrads[:,:,obj], outXGrad[:,:,channel_list[ch]])
				outYGrad[:,:,channel_list[ch]]=np.where(masks[:,:,obj]==True, ygrads[:,:,obj], outYGrad[:,:,channel_list[ch]])
				placed = True
			else:
				ch += 1
		if ch == outCH and not placed:
			#issue with needing more channels than just 3!
			print("Need more than {} channels for output... Pausing for debugger.".format(outCH))
			import pdb;pdb.set_trace()

	outXGrad = outXGrad.astype(np.float32)
	outYGrad = outYGrad.astype(np.float32)
	return outMasks, outBorders, outXGrad, outYGrad

def _project_all_by_Z(masks, zclasses, outCH):
	"""
	V1.6 New attempt CE: 01/13/22
	This places objects into channels based on their classification (zclasses).
	Can still have overlap between cells, however.
	"""
	H, W, N = masks.shape
	if len(zclasses)!=N:
		print("pausing different number of objects in masks and zclasses...")
		import pdb;pdb.set_trace()
	#sort masks by zclasses first!
	# this way, if cells are added to the same image, then they are added based
	# on their z location.
	sortinds = np.argsort(zclasses)
	masks = masks[:,:,sortinds]
	zclasses = zclasses[sortinds]

	#load borders
	borders = create_borders_N(masks)
	_,_,Nborders = borders.shape
	if N != Nborders:
		print("pausing different number of objects...")
		import pdb;pdb.set_trace()

	outMasks = np.zeros(shape=(H,W,outCH), dtype=masks.dtype)
	outBorders = np.zeros(shape=(H,W,outCH), dtype=borders.dtype)
	channel_list = list(range(outCH))

	#load gradients
	centers,niters,bboxes = _find_centers(masks)
	xgrads,ygrads = _run_gradient_find(centers,niters,bboxes,masks)

	outXGrad = np.zeros(shape=(H,W,outCH), dtype=xgrads.dtype)
	outYGrad = np.zeros(shape=(H,W,outCH), dtype=ygrads.dtype)
	#place objects in ground truth outputs randomly, but they should be in the same channel.

	for obj in range(N):
		P = np.argwhere(masks[:,:,obj]==True)
		outMasks[:,:,zclasses[obj]]=np.where(masks[:,:,obj]==True, masks[:,:,obj], outMasks[:,:,zclasses[obj]])
		outBorders[:,:,zclasses[obj]]=np.where(borders[:,:,obj]==True, borders[:,:,obj], outBorders[:,:,zclasses[obj]])
		outXGrad[:,:,zclasses[obj]]=np.where(masks[:,:,obj]==True, xgrads[:,:,obj], outXGrad[:,:,zclasses[obj]])
		outYGrad[:,:,zclasses[obj]]=np.where(masks[:,:,obj]==True, ygrads[:,:,obj], outYGrad[:,:,zclasses[obj]])

	outXGrad = outXGrad.astype(np.float32)
	outYGrad = outYGrad.astype(np.float32)
	return outMasks, outBorders, outXGrad, outYGrad

"""
The best method would be to take the argmax locations when doing the maximum
projection of the actual image, then the gradients should match closely to that.
The problem is actually that the slice numbers of the image do not line up with
the slice numbers of the grads [grads has object numbers, image has z slices]
So what to do?
"""

def _run_gradient_find(centers, niters, bboxes, masks):
	xgrads = np.zeros(shape=masks.shape).astype(np.float32)
	ygrads = np.zeros(shape=masks.shape).astype(np.float32)
	#so, some objects might be overlapping...
	#run heat diffusion on each mask.
	#then Add all together. Those with multiple masks will need to be looked at carefully. I wonder what they will show?
	s_unit = np.ones((3,3),dtype=np.float32)/9.0
	change_unit = 1e8
	for i in range(centers.shape[0]):
		m = masks[bboxes[i,1]:bboxes[i,3],bboxes[i,0]:bboxes[i,2],i] #boolean array
		yy,xx=np.nonzero(m)
		xx=np.unique(xx)
		yy=np.unique(yy)
		heat_map = np.zeros(shape=m.shape)

		#first, make sure the "center" point is indeed a value of 1.
		if m[centers[i,1] - bboxes[i,1],centers[i,0]-bboxes[i,0]]!=True:
			print("Likely the centers are in form x-y and should be y-x")
			import pdb;pdb.set_trace()
		assert m[centers[i,1] - bboxes[i,1],centers[i,0]-bboxes[i,0]]==True, "Likely the centers are in form x-y and should be y-x"
		#add 1 to the center.
		for quart in range(4):
			for n in range(int(np.ceil(niters[i]/4))):
				heat_map[centers[i,1] - bboxes[i,1],centers[i,0]-bboxes[i,0]]+=(change_unit**quart)/10.
				#Much, much faster doing it this way.
				#convolve with s_unit
				heat_map = scipy.signal.convolve2d(heat_map,s_unit,mode='same')
				heat_map = heat_map * m
			if quart<3:
				heat_map = np.where(m>0,heat_map*change_unit,0.0)
		#this addition was arbitrary. I noticed that if the cell is really long then the
		#iterations never really reach the end of the mask. Therefore, to help it along,
		#I add a bit. Maybe this value is too big though?
		# for n in range(int(np.ceil(3*niters[i]/4))):
		# 	heat_map[centers[i,1] - bboxes[i,1],centers[i,0]-bboxes[i,0]]+=1
		# 	#Much, much faster doing it this way.
		# 	#convolve with s_unit
		# 	heat_map = scipy.signal.convolve2d(heat_map,s_unit,mode='same')
		# 	heat_map = heat_map * m
		#this doesn't work.


		#heat_map = np.log(1+((1000/np.min(heat_map[m==True]))*heat_map))
		#heat_map = np.where(heat_map>0,np.log(1+1e-15+heat_map),0.0)
		#So the big issue is the values could be near the same.
		#That is, any value of size < 1e-15 essentially is not calculated well.
		#ex. np.log(np.exp(1+3e-16))-np.log(np.exp(1+2e-16))
		#log has a rounding issue, assumes float32 rather than float64
		heat_map = np.log(1+heat_map)

		xgrad = _run_gradient_x(heat_map)
		ygrad = _run_gradient_y(heat_map)
		grads = np.stack((xgrad,ygrad))
		grads /= (1e-20 + np.sqrt(np.sum((grads**2),axis=0)))
		if np.isnan(np.sum(grads[0])):
			print("xgrad problem")
			import pdb;pdb.set_trace()
		if np.isnan(np.sum(grads[1])):
			print("ygrad problem")
			import pdb;pdb.set_trace()
		#import pdb;pdb.set_trace()
		xgrads[bboxes[i,1]:bboxes[i,3],bboxes[i,0]:bboxes[i,2],i]=grads[0]
		ygrads[bboxes[i,1]:bboxes[i,3],bboxes[i,0]:bboxes[i,2],i]=grads[1]
		# import pdb;pdb.set_trace()
		# fig,(ax1,ax2,ax3)=plt.subplots(1,3)
		# ax1.imshow(m)
		# ax2.imshow(grads[0])
		# ax3.imshow(grads[1])
		# plt.show()
	return xgrads, ygrads

def _run_gradient_x(heat_map):
	s_unit = np.array([[0,0,0],[-1,0,1],[0,0,0]])/2
	x_grad = scipy.signal.convolve2d(heat_map,s_unit,mode='same')
	#if np.max(np.abs(x_grad))>0:
	#	x_grad = x_grad / np.max(np.abs(np.ravel(x_grad)))
	return x_grad

def _run_gradient_y(heat_map):
	s_unit = np.array([[0,-1,0],[0,0,0],[0,1,0]])/2
	y_grad = scipy.signal.convolve2d(heat_map,s_unit,mode='same')
	#if np.max(np.abs(y_grad))>):
	#	y_grad = y_grad / np.max(np.abs(np.ravel(y_grad)))
	return y_grad


def _find_centers(masks, mode="median", area_thresh=25):
	"""
	Need to add a new mode that guarentees center will be in the object.
	masks is hxwxN
	"""
	assert mode in ["median","bbox","centroid"], "mode argument can only be median or centroid"
	bboxes= np.array([np.flip(np.reshape(r.bbox,[2,2]),axis=1).flatten().tolist() for num_cell in range(masks.shape[2]) for i,r in enumerate(skimage.measure.regionprops(masks[:,:,num_cell].astype(int)))],dtype='int32')
	if mode=="bbox":
		#use bounding box to find middle.
		#(min_row, min_col, max_row, max_col)
		#if you do plt imshow, it flips it so remember in that case it is (min_col, min_row, max_col, max_row)
		centers = np.matrix.round(bboxes[:,[0,1]]+(bboxes[:,[2,3]]-bboxes[:,[0,1]])/2).astype(int)
		print("WARNING: Using bbox argument may put centers not on cell body, \n\
		as the bounding box is used to determine approximate cell center")
		#returns the N x 2 array as integers
	elif mode=="centroid":
		#return integer centroids
		centroids= np.array([list(r.centroid[::-1]) for num_cell in range(masks.shape[2]) for i,r in enumerate(skimage.measure.regionprops(masks[:,:,num_cell].astype(int)))],dtype='int32')
		centers = np.matrix.round(centroids).astype(int)
		print("WARNING: Using median argument may put centers not on cell body, \n\
		as the bounding box is used to determine approximate cell center")

	elif mode=="median":
		#for each one, we need the median.
		centers = np.zeros((masks.shape[2],2))
		for i in range(masks.shape[2]):
			y,x=np.nonzero(masks[:,:,i])
			#find median
			centers[i,0] = int(np.median(x))
			centers[i,1] = int(np.median(y))
			if masks[int(centers[i,1]),int(centers[i,0]),i]!=True:
				#we need to correctly find the closest nonzero pixel.
				#find dinstance from centers to nonzero pixels. easy.
				D = np.sqrt(np.sum(((x-centers[i,0])**2,(y-centers[i,1])**2),axis=0))
				centers[i,0]=x[np.argmin(D)]
				centers[i,1]=y[np.argmin(D)]
		centers=centers.astype(int)

	if centers.shape[0]!=bboxes.shape[0]:
		import pdb;pdb.set_trace()
	niters = np.ceil(4.*np.sqrt(np.sum(np.square(bboxes[:,[2,3]]-bboxes[:,[0,1]]),axis=1))).astype(int)
	#diagonal length, times 2.
	#niters = np.matrix.round(2*np.sum(bboxes[:,[2,3]]-bboxes[:,[0,1]],axis=1)).astype(int)
	#import pdb;pdb.set_trace()
	#this returns an N x 1 matrix. Hopefully. Or 1 x N, whatever.
	#You need to make sure that centers from bboxes and centers from centroids are about the same. Otherwise, you need to change the bbox code above.
	#Verified, CE.
	return centers, niters, bboxes

def plt_show_many(M, out=2):
	if out==2:
		fig,(ax1,ax2) = plt.subplots(1,2)
		ax1.imshow(M[:,:,0])
		ax2.imshow(M[:,:,1])
		plt.show()
	elif out==3:
		fig,(ax1,ax2,ax3) = plt.subplots(1,3)
		ax1.imshow(M[:,:,0])
		ax2.imshow(M[:,:,1])
		ax3.imshow(M[:,:,2])
		plt.show()
	elif out==4:
		fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
		ax1.imshow(M[:,:,0])
		ax2.imshow(M[:,:,1])
		ax3.imshow(M[:,:,2])
		ax4.imshow(M[:,:,3])
		plt.show()
	elif out==5:
		fig,((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3)
		ax1.imshow(M[:,:,0])
		ax2.imshow(M[:,:,1])
		ax3.imshow(M[:,:,2])
		ax4.imshow(M[:,:,3])
		ax5.imshow(M[:,:,4])
		plt.show()
	elif out==6:
		fig,((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3)
		ax1.imshow(M[:,:,0])
		ax2.imshow(M[:,:,1])
		ax3.imshow(M[:,:,2])
		ax4.imshow(M[:,:,3])
		ax5.imshow(M[:,:,4])
		ax6.imshow(M[:,:,5])
		plt.show()
	else:
		print("Need more options!")
		import pdb;pdb.set_trace()

def determine_obj_z(zstack, masks):
	"""
	INPUTS
	-------------
	zstack = H x W x Z x 1 numpy float array, Z-stack of gray images.
	masks = H x W x N numpy boolean array, where each channel contains only one object.

	OUTPUTS
	-------------
	obj_z_locs = N numpy integer array with approximate z location.

	WARNINGS
	-------------
	Fails if many objects are closely packed.
	"""
	obj_z_locs = np.zeros(masks.shape[2],dtype=int)
	obj_locs = np.zeros((masks.shape[2],3))
	for obj in range(masks.shape[2]):
		inds = np.argwhere(masks[:,:,obj]==True)
		#inds is an Nx2 array.
		zprofile = [np.mean([zstack[r,c,slice,0] for r,c in inds]) for slice in range(zstack.shape[2])]
		obj_z_locs[obj] = np.argmax(zprofile)
		obj_locs[obj,2] = np.argmax(zprofile)
		obj_locs[obj,0] = np.mean(inds[:,0])
		obj_locs[obj,1] = np.mean(inds[:,1])
	return obj_z_locs, obj_locs

####PLOTTING FOR OPTIMAL PADDING DETERMINATION####
def plot_padding(padding_options, images):
	"""
	INPUTS
	----------------------
	padding_options = list, contains integers representing how many padded slices
					are added BEFORE the image sequence (Note, to fill to the
					proper z-width, padded slices are also added at the end.)
	images = np.array with dimensions B x H x W x 3, predicted images from PEN module.
	"""
	from matplotlib.widgets import Slider, Button
	import copy
	# Create the figure and the line that we will manipulate
	fig, ax = plt.subplots()
	shown_image = ax.imshow(images[0])
	#use shown_image.set_data(images[z])
	# adjust the main plot to make room for the sliders
	plt.subplots_adjust(bottom=0.25)
	# Make a horizontal slider to control the initial padding
	axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
	valinit = padding_options[len(padding_options)//2]
	slider_value = copy.copy(valinit)
	# define the values to use for snapping
	allowed_padding = np.array(padding_options)
	slider_handle = Slider(ax=axfreq,label='Initial Padding', valmin=padding_options[0], \
		valmax=padding_options[-1], valinit=valinit, valstep=1)

	def update(val):
		#import pdb;pdb.set_trace()
		global slider_value
		slider_value = val
		if isinstance(val,int):
			shown_image.set_data(images[val])
		else:
			shown_image.set_data(images[val.astype(int)])
		fig.canvas.draw_idle()

	slider_handle.on_changed(update)

	ax_reset = plt.axes([0.8, 0.025, 0.1, 0.04])
	ax_save = plt.axes([0.6, 0.025, 0.15, 0.04])
	button = Button(ax_reset, 'Reset', hovercolor='0.975')
	button_close = Button(ax_save, 'Select Padding', hovercolor='0.975')

	def reset(event):
		slider_handle.reset()

	def close(event):
		plt.close()

	button.on_clicked(reset)
	button_close.on_clicked(close)
	plt.show()
	print("You selected the best padding strategy with {} in the beginning of the image.".format(slider_handle.val))
	return slider_handle.val

### LOAD CONFIG SETTINGS IF AVAILABLE.
def load_config_file(fpath, config):
	if not os.path.isfile(fpath):
		print("\n README.txt config file does not exist in weights path. Proceeding with default configuration...\n")
	else:
		import ast
		import re
		README_dict = {}
		with open(fpath) as file:
			lines = [line.rstrip() for line in file]
		begin=False

		for i,line in enumerate(lines):
			if line=='CONFIGURATION SETTINGS:':
				#want to begin on the NEXT line.
				begin=True
				continue

			if begin==1:
				(key, val) = line.split(maxsplit=1)
				try:
					#anything that is not MEANT to be a string.
					#mostly this does fine on its own.
					README_dict[key] = ast.literal_eval(val)
				except:
					try:
						#messes up list attributes where there spaces are not uniform sometimes.
						README_dict[key] = ast.literal_eval(re.sub("\s+", ",", val.strip()))
					except:
						README_dict[key] = val

		print("\n Replacing default config key values with previous model's config file... \n")
		for func in dir(config):
			if not func.startswith("__") and not callable(getattr(config, func)):
				#print("{:30} {}".format(a, getattr(self, a)))
				if func in README_dict.keys():
					#special case if it is a dictionary.
					if isinstance(README_dict[func],dict):
						#change keys if they exist in config.
						#get the dictionary from config.
						config_dict = getattr(config, func)
						#change values.
						#get keys in README_dict[func]
						for RM_key in README_dict[func].keys():
							#if RM_key in config_dict.keys():
							#	#reset value.
							config_dict[RM_key] = README_dict[func][RM_key]
							#else:
							#	#add new key...
						#set into config.
						setattr(config, func, config_dict)
					else:
						setattr(config, func, README_dict[func])

	return config

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

# def stack_augmentation_fast(image_a, image_b, mask_a, mask_b, zlocs_a, zlocs_b, centroids_a, centroids_b, final_z_size):
# 	#VERY SLOW, need to eliminate the deep copies!
# 	image_z_size = image_a.shape[2]
# 	for f in range(int(np.floor(final_z_size/ image_a.shape[2]))-1):
# 		if np.random.rand()>=0.5:
# 			#flip with z
# 			image_b, zlocs_b = image_zflip(image_b, image_z_size, zlocs_b)
#
# 		zlocs_b += image_z_size
# 		#rotate?
# 		flag = True
# 		while flag:
# 			image_b, mask_b, centroids_b, flag = image_mask_rotate(image_b, mask_b, centroids_b, 0.5)
# 		image_b, mask_b, centroids_b, _ = image_xyflip(image_b, mask_b, centroids_b, 0.5)
#
# 		image_a, mask_a, _ = stack_image(image_a, image_b, mask_a, mask_b, z_to = final_z_size)
# 		#append centroids
# 		centroids_a = np.concatenate([centroids_a,centroids_b])
# 		#append z-locs
# 		zlocs_a = np.concatenate([zlocs_a, zlocs_b])
#
# 	return image_a, mask_a, zlocs_a, centroids_a

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


# def dense_augmentation(image, mask, zlocs, centroids):
# 	#add probability
# 	# if np.random.rand()>=0.2:
# 	image_og = copy.deepcopy(image)
# 	im_z_size = image_og.shape[2]
# 	mask_og = copy.deepcopy(mask)
# 	#z flip and rotate
# 	zflag=False
# 	if np.random.rand()>=0.5:
# 		image, znew = image_zflip(image, im_z_size, zlocs)
# 		zflag=True
# 	else:
# 		znew = zlocs
# 	#if we flip it, we need to maintain that info for the mask.
# 	#rotate!
# 	image, mask, centnew, rotflag = image_mask_rotate(image, mask, centroids, 0.5)
# 	#flip!
# 	image, mask, centnew, xyflag = image_xyflip(image, mask, centnew, 0.5)
# 	#
# 	if np.any([rotflag, xyflag, zflag]):
# 		image = np.where(image>image_og, image, image_og)
# 		mask = np.concatenate([mask_og, mask], axis=2)
# 		#append centroids
# 		centroids = np.concatenate([centroids,centnew])
# 		#append z-locs
# 		#import pdb;pdb.set_trace()
# 		zlocs = np.concatenate([zlocs, znew])
# 		return image, mask, zlocs, centroids
# 	else:
# 		#the image was not changed at all. In this case, we don't want to add anything
# 		return image_og, mask_og, zlocs, centroids

def dense_augmentation(image, mask, zlocs, centroids):
	"""
	Do the same operations as dense_augmentation, but we'll also add a translation.
	"""
	image_og = copy.deepcopy(image)
	im_z_size = image_og.shape[2]
	mask_og = copy.deepcopy(mask)
	"""
	Change the second stack
	"""
	#z flip and rotate
	image, znew, zflag1 = image_zflip(image, im_z_size, zlocs, 0.5)
	#if we flip it, we need to maintain that info for the mask.
	#rotate!
	image, mask, centnew, rotflag1 = image_mask_rotate(image, mask, centroids, 0.5)
	#flip!
	image, mask, centnew, xyflag1 = image_xyflip(image, mask, centnew, 0.5)
	#
	"""
	Change the first stack!
	"""
	#z flip and rotate
	image_og, zlocs, zflag2 = image_zflip(image_og, im_z_size, zlocs, 0.5)
	#if we flip it, we need to maintain that info for the mask.
	#rotate!
	image_og, mask_og, centroids, rotflag2 = image_mask_rotate(image_og, mask_og, centroids, 0.5)
	#flip!
	image_og, mask_og, centroids, xyflag2 = image_xyflip(image_og, mask_og, centroids, 0.5)

	if np.logical_and(np.any([rotflag1, xyflag1, zflag1, rotflag2, xyflag2, zflag2]), np.logical_or(~np.logical_and(zflag1, zflag2), np.any([rotflag1, xyflag1, rotflag2, xyflag2]))):
		image = np.where(image>image_og, image, image_og)
		mask = np.concatenate([mask_og, mask], axis=2)
		#append centroids
		centroids = np.concatenate([centroids,centnew])
		#append z-locs
		#import pdb;pdb.set_trace()
		zlocs = np.concatenate([zlocs, znew])
		return image, mask, zlocs, centroids
	else:
		#the image was not changed at all. In this case, we don't want to add anything
		return image_og, mask_og, zlocs, centroids


def dense_translate_augmentation(image, mask, zlocs, centroids, final_z_size, N=1, min_prob=0.5):
	"""
	Do the same operations as dense_augmentation, but we'll also add a translation.

	we would like to do this operation not just on one stack, but repeat the operation
	on several stacks.
	In fact, we should repeat the process as many as config.OUT_CHANNELS to make sure
	that it could properly separate cells into at least that many overlapping color channels.
	"""
	image_og = copy.deepcopy(image)
	im_z_size = image_og.shape[2]
	pad_z_amount = final_z_size - im_z_size
	# if pad_z_amount>im_z_size:
	# 	pad_z_amount = image_og.shape[2]
	#print("pad_z = {}".format(pad_z_amount))
	mask_og = copy.deepcopy(mask)

	"""
	Change the first image, call it IM.
	For every subsequent n in range(N), add it to IM.
	"""
	image_og = copy.deepcopy(image)
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
			#import pdb;pdb.set_trace()
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
# """
# Change the secondary stacks
# """
# #z flip and rotate
# image, znew, zflag1 = image_zflip(image, im_z_size, zlocs, min_prob)
# #pick any number between pad_z_amount and 0.
# pad_z = int(np.round(np.random.rand()*pad_z_amount))
# znew += pad_z
# #if we flip it, we need to maintain that info for the mask.
# #rotate!
# image, mask, centnew, rotflag1 = image_mask_rotate(image, mask, centroids, min_prob)
# #flip!
# image, mask, centnew, xyflag1 = image_xyflip(image, mask, centnew, min_prob)
# #
# """
# Change the first stack!
# """
# #z flip and rotate
# image_og, zlocs, zflag2 = image_zflip(image_og, im_z_size, zlocs, min_prob)
# #if we flip it, we need to maintain that info for the mask.
# #rotate!
# image_og, mask_og, centroids, rotflag2 = image_mask_rotate(image_og, mask_og, centroids, min_prob)
# #flip!
# image_og, mask_og, centroids, xyflag2 = image_xyflip(image_og, mask_og, centroids, min_prob)
#
# if np.logical_or(zflag1!=zflag1,  np.any([rotflag1, xyflag1, rotflag2, xyflag2])):
# 	#np.logical_and(np.any([rotflag1, xyflag1, zflag1, rotflag2, xyflag2, zflag2]), np.logical_or(~np.logical_and(zflag1, zflag2), np.any([rotflag1, xyflag1, rotflag2, xyflag2]))):
# 	image_og = np.pad(image_og, ((0,0),(0,0),(0,pad_z),(0,0)), 'constant')
# 	image = np.pad(image, ((0,0),(0,0),(pad_z,0),(0,0)), 'constant')
# 	image = np.where(image>image_og, image, image_og)
# 	mask = np.concatenate([mask_og, mask], axis=2)
# 	#append centroids
# 	centroids = np.concatenate([centroids,centnew])
# 	#append z-locs
# 	#import pdb;pdb.set_trace()
# 	zlocs = np.concatenate([zlocs, znew])
# 	return image, mask, zlocs, centroids
# else:
# 	#the image was not changed at all. In this case, we don't want to add anything
# 	return image_og, mask_og, zlocs, centroids

############################################################################
############################################################################
############################################################################
############################################################################

# def dense_translate_augmentation_fast(image_a, image_b, mask_a, mask_b, zlocs_a, zlocs_b, centroids_a, centroids_b, final_z_size, min_prob=0.5):
# 	"""
# 	Do the same operations as dense_augmentation, but we'll also add a translation.
#	Turns out to not change the computation time too much.
# 	"""
# 	#eliminate the deep copies....
# 	im_z_size = image_a.shape[2]
# 	pad_z_amount = final_z_size - im_z_size
# 	if pad_z_amount>im_z_size:
# 		pad_z_amount = image_a.shape[2]
# 	#pick any number between pad_z_amount and 0.
# 	pad_z_amount = int(np.round(np.random.rand()*pad_z_amount))
# 	"""
# 	Change the second stack
# 	"""
# 	#z flip and rotate
# 	zflag1=False
# 	if np.random.rand()>=min_prob:
# 		image_b, zlocs_b = image_zflip(image_b, im_z_size, zlocs_b)
# 		zflag1=True
#
# 	zlocs_b += pad_z_amount
# 	#if we flip it, we need to maintain that info for the mask.
# 	#rotate!
# 	image_b, mask_b, centroids_b, rotflag1 = image_mask_rotate(image_b, mask_b, centroids_b, min_prob)
# 	#flip!
# 	image_b, mask_b, centroids_b, xyflag1 = image_xyflip(image_b, mask_b, centroids_b, min_prob)
# 	#
# 	"""
# 	Change the first stack!
# 	"""
# 	#z flip and rotate
# 	zflag2=False
# 	if np.random.rand()>=min_prob:
# 		image_a, zlocs_a = image_zflip(image_a, im_z_size, zlocs_a)
# 		zflag2=True
# 	#if we flip it, we need to maintain that info for the mask.
# 	#rotate!
# 	image_a, mask_a, centroids_a, rotflag2 = image_mask_rotate(image_a, mask_a, centroids_a, min_prob)
# 	#flip!
# 	image_a, mask_a, centroids_a, xyflag2 = image_xyflip(image_a, mask_a, centroids_a, min_prob)
#
# 	if np.logical_and(np.any([rotflag1, xyflag1, zflag1, rotflag2, xyflag2, zflag2]), np.logical_or(~np.logical_and(zflag1, zflag2), np.any([rotflag1, xyflag1, rotflag2, xyflag2]))):
# 		image_a = np.pad(image_a, ((0,0),(0,0),(0,pad_z_amount),(0,0)), 'constant')
# 		image_b = np.pad(image_b, ((0,0),(0,0),(pad_z_amount,0),(0,0)), 'constant')
# 		image_b = np.where(image_b>image_a, image_b, image_a)
# 		mask_b = np.concatenate([mask_a, mask_b], axis=2)
# 		#append centroids
# 		centroids_a = np.concatenate([centroids_a,centroids_b])
# 		#append z-locs
# 		zlocs_a = np.concatenate([zlocs_a, zlocs_b])
# 		return image_b, mask_b, zlocs_a, centroids_a
# 	else:
# 		#the image was not changed at all. In this case, we don't want to add anything
# 		return image_a, mask_a, zlocs_a, centroids_a

################################################################################
################################################################################

def linearly_color_encode_image(image, out_ch = 3):
	"""
	INPUTS
	-----------------------------------------
	image = np.array, dimensionality of H x W x Z x 1, dtype float32
			input Z-stack image array AFTER APPLYING APPROPRIATE PADDING.

	out_ch = integer, specifies number of output channels to mimic what PEN
			does operationally. Recommended is 3, this is the typical channel
			input for deep learning computer vision networks.

	RETURNS
	------------------------------------------
	output = np.array, dimensionality of H x W x 3, dtype = same as input image.
			linearly colorized z-encoding of image. max projection of color
			channels over each pixel.

	"""
	#image needs to be the padded verison.
	#image has shape [H,W,Z,1]
	#if it unpadded, then we also need to know the padding strategy; not a good idea.
	z_slices = image.shape[2]
	space = z_slices/(out_ch+1) #space between the peaks of each output channel.
	#space = (z_slices-1)/max(1,out_ch-1)#(out_ch-1)
	#centers = [0] + [(i+1) * space for i in range(int(out_ch-1))]
	centers = [(i+1) * space for i in range(int(out_ch))] #calculate the center locations of each color Gaussian.
	midway_overlap_percentage = 0.5 #between each peak, specify the degree of each color influence.
	sigma = np.sqrt(-0.25*(space**2) / (2*np.log(midway_overlap_percentage)))
	#calculates the appropriate sigma to make the midway_overlap_percentage true.
	xr = np.linspace(0,z_slices-1,z_slices)
	G = np.exp(-((xr - np.expand_dims(centers,axis=-1))**2)/(2*(sigma**2)))
	#G has shape out_ch x z_slices
	G = np.expand_dims(np.expand_dims(np.rollaxis(G,0,2),axis=0),axis=0)
	#G has shape 1 x 1 x z_slices x out_ch
	#tested in Linearly_Encode.py
	#multiplying image*G returns array shape H x W x Z x 3
	#take max over each color!
	return np.max(image*G, axis=2) #has shape H x W x out_ch
