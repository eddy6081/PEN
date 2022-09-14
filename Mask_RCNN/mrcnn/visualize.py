"""
AUTHOR: CHRISTOPHER Z. EDDY
CONTACT: eddych@oregonstate.edu
Purpose: visualization of dataset for Mask-RCNN segmentation.
"""

import numpy as np
import matplotlib.patches as patches
#import matplotlib.pyplot as plt
import json
import colorsys
import random
from skimage import io, exposure, draw
from skimage.measure import find_contours
import random, os

import sys
import matplotlib.pyplot as plt


#############################################################################
#############################DEFINE FUNCTIONS################################
#############################################################################

def random_colors(N, bright=True):
	"""
	Generate random colors.
	To get visually distinct colors, generate them in HSV space then
	convert to RGB.
	"""
	brightness = 1.0 if bright else 0.7
	hsv = [(i / N, 1, brightness) for i in range(N)]
	colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
	random.shuffle(colors)
	return colors

def log_transform_image(img):
	# Logarithmic
	if np.max(img)<=1:
		rescale=True
		img = img * 255 / np.max(img)
		img = img+1;
		img[img>255]=255.0
		#make min zero, make max 1
	else:
		rescale=False
		img = img+1;
		img[img>255]=255.0
	logarithmic_corrected = np.log(img)*255/np.log(255)
	logarithmic_corrected = logarithmic_corrected.astype(np.float32)
	#exposure.adjust_log(img, 1)
	if rescale:
		img = img / 255
		img = img / np.max(img)
	return logarithmic_corrected

def apply_mask(image, mask, color, alpha=0.2):
	"""Apply the given mask to the image.
	"""
	if image.dtype=="float32":
		for c in range(3):
			image[:, :, c] = np.where(mask == 1,
									  image[:, :, c] *
									  (1 - alpha) + alpha * color[c] * 1,
									  image[:, :, c])
	else:
		for c in range(3):
			image[:, :, c] = np.where(mask == 1,
									  image[:, :, c] *
									  (1 - alpha) + alpha * color[c] * 255,
									  image[:, :, c])
	return image

def show_max_proj(image,transform=True,ax=None):
	"""
	PURPOSE: Display maximum projection from z-stack tif file.
	INPUTS:
	image = z-stack tif file
	transform = boolean of whether to apply log transform on max projection
	"""
	#if not ax:
	#	_, ax = plt.subplots(1)
	if len(image.shape)>3:
		IM_MAX = np.max(image, axis=2)
		if len(IM_MAX.shape)==3:
			IM_MAX = np.squeeze(IM_MAX,axis=2)
		if transform:
			IM_MAX = log_transform_image(IM_MAX)
		# Display the image
		IM_MAX = np.stack((IM_MAX,)*3,axis=-1)
		#normalize to range 0..1
		IM_MAX = IM_MAX / np.max(IM_MAX)
	else:
		IM_MAX = image
	#ax.axis('off')
	#ax.imshow(IM_MAX)
	return IM_MAX

def binary_mask_to_polygon_skimage(mask):
	"""
	mask should have shape H x W x num_instances with one instance per channel.
	mask comes from dataset.load_mask(num).
	"""
	polygons_x = []
	polygons_y = []
	#a=[]
	for num_inst in range(mask.shape[2]):
		binary_mask = mask[:,:,num_inst]
		#we want to pad binary_mask one on each side. Then subtract the same pad from each.
		if binary_mask.dtype=='bool':
			binary_mask = np.pad(binary_mask,((1,1),(1,1)),constant_values=(False,False))
		else:
			binary_mask = np.pad(binary_mask,((1,1),(1,1)),constant_values=(0,0))
		contours = find_contours(binary_mask, 0.5, fully_connected='high') #see documentation for 0.5
		for contour in contours:
			contour = np.flip(contour, axis=1)
			if len(contour) < 3:
				continue
			segmentation_x = contour[:,0].tolist()
			segmentation_y = contour[:,1].tolist()
			#contour.ravel().tolist()
			segmentation_x = [0 if i-1 < 0 else i-1 for i in segmentation_x]
			segmentation_y = [0 if i-1 < 0 else i-1 for i in segmentation_y]
			# after padding and subtracting 1 we may get -0.5 points in our segmentation
			#if the threshold area is too low, do not include it
			polygons_x.append(segmentation_x)
			polygons_y.append(segmentation_y)
			#a.append(PolyArea(segmentation_x,segmentation_y))
	return [polygons_x,polygons_y]#,a]

def display_training_instances(image, mask, ax=None):
	#display image
	auto_show=False
	if not ax:
		auto_show=True
		fig,ax = plt.subplots(1)

	IM = show_max_proj(image,transform=False)
	#IM = np.squeeze(IM,axis=2)
	[px, py] = binary_mask_to_polygon_skimage(mask)
	colors = random_colors(len(px))

	#draw polygons and IM on ax.
	ax.imshow(IM)
	#import pdb;pdb.set_trace()
	for i,(x,y) in enumerate(zip(px,py)):
		# Subtract the padding and flip (y, x) to (x, y)
		verts = np.stack((np.array(x),np.array(y)),axis=1)
		p = patches.Polygon(verts, facecolor="none", edgecolor=colors[i], alpha=0.6)
		ax.add_patch(p)
		p = patches.Polygon(verts, facecolor=colors[i], edgecolor="none", alpha=0.2)
		ax.add_patch(p)
	if auto_show:
		plt.show()

class PPlot():
	"""
	Generates an interactive plot which accepts keystrokes "right arrow" and "escape"
	Shows training images and mask annotations
	"""
	def __init__(self, cell_dataset, config, image_ids):
		self.image = None
		self.mask = None
		self.xl = None
		self.ttl = None
		self.image_id = image_ids #can be a list or an integer.
		self.image_id_i = 0
		self.fig, self.ax = plt.subplots(1,figsize=(12,10))
		self.cid = self.fig.canvas.mpl_connect('key_press_event', self.press)
		self.config = config
		self.dataset=cell_dataset
		self.update()
		print("\n        *********OPTIONAL COMMANDS*********\n\
        'right arrow' to see next image \n\
        'left arrow' to see previous image\n\
        'ESC' to exit\n\
        ***********************************\n")
		plt.show()
	def update(self):
		self.ax.clear()
		self.mask,_ = self.dataset.load_mask(self.image_id[self.image_id_i])
		self.image = self.dataset.load_image(self.image_id[self.image_id_i], dimensionality="3D", mask=self.mask, z_to = self.config.INPUT_Z)
		display_training_instances(self.image, self.mask, self.ax)
		#below assumes that the output always has a third channel. If there is only one detection, does this work?
		self.xl = "{} detections".format(self.mask.shape[2])
		self.ttl = "{}, id {}".format(self.dataset.image_info[self.image_id[self.image_id_i]]['id'],self.image_id[self.image_id_i])
		self.ax.set_xlabel(self.xl)
		self.ax.set_title(self.ttl)
		self.fig.canvas.draw()
		plt.pause(1*10e-10)
	def press(self, event):
		print('press', event.key)
		sys.stdout.flush()
		if event.key == 'right':
			self.image_id_i += 1
			if self.image_id_i<len(self.image_id):
				self.update()
			else:
				print('you have reviewed all images, press ESC to continue')
				self.image_id_i=len(self.image_id)-1
		if event.key == 'left':
			self.image_id_i -= 1
			if self.image_id_i>=0:
				self.update()
			else:
				print('cannot go back')
				self.image_id_i=0
		elif event.key == 'escape':
			print('quitting')
			self.fig.canvas.mpl_disconnect(self.cid)
			plt.close(self.fig)
			return

def show_train_set(config, cell_dataset, n=1, image_id=None):
	"""
	n = number of training images to show.
	config from cellconfig
	cell_dataset from CellDataset
	"""
	if image_id:
		mask,_ = cell_dataset.load_mask(image_id)
		im = cell_dataset.load_image(image_id, dimensionality="3D", mask=mask, z_to = config.INPUT_Z)
		display_training_instances(im, mask)
	else:
		if n < len(cell_dataset.image_info):
			#np.random.seed(19680801)
			image_ids = np.random.choice(list(range(len(cell_dataset.image_info))),size=n, replace=False)
			p = PPlot(cell_dataset,config,image_ids)
		else:
			image_ids = list(range(len(cell_dataset.image_info)))
			p = PPlot(cell_dataset,config,image_ids)

# def display_instances_tiff(image, boxes, masks, class_ids, class_names,
# 					  scores=None,
# 					  show_mask=True, show_bbox=True,
# 					  colors=None, captions=None):
# 		"""
# 		image is RGB
# 		boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
# 		masks: [height, width, num_instances]
# 		class_ids: [num_instances]
# 		class_names: list of class names of the dataset
# 		scores: (optional) confidence scores for each box
# 		show_mask, show_bbox: To show masks and bounding boxes or not
# 		colors: (optional) An array or colors to use with each object
# 		captions: (optional) A list of strings to use as captions for each object
# 		"""
# 	# Number of instances
# 	N = boxes.shape[0]
# 	if not N:
# 		print("\n*** No instances to display *** \n")
# 	else:
# 		assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
#
# 	# Generate random colors
# 	colors = colors or random_colors(N)
#
# 	masked_image = image
# 	for i in range(N):
# 		color = colors[i]
#
# 		# Bounding box
# 		if not np.any(boxes[i]):
# 			# Skip this instance. Has no bbox. Likely lost in image cropping.
# 			continue
# 		y1, x1, y2, x2 = boxes[i]
# 		wmap = np.zeros(shape=(image.shape[0],image.shape[1],1))
# 		if show_bbox:
# 			#make nx2 vertices.
# 			poly = np.array([[x1,y1],[x1,y2],[x2,y2],[x2,y1]])
# 			rr, cc = draw.polygon(poly[:,0], poly[:,1], wmap.shape[0:-1],fill=0)
# 			wmap[rr,cc,0] = 1
#
# 		masked_image=np.where(annotation_map==1,0.0,masked_image)
#
#
# 		# Label
# 		if not captions:
# 			class_id = class_ids[i]
# 			score = scores[i] if scores is not None else None
# 			label = class_names[class_id]
# 			caption = "{} {:.3f}".format(label, score) if score else label
# 		else:
# 			caption = captions[i]
# 		ax.text(x1, y1, caption,
# 				color='w', size=5, backgroundcolor="none") # was y1+8
#
# 		# Mask
# 		mask = masks[:, :, i]
# 		if show_mask:
# 			masked_image = apply_mask(masked_image, mask, color)
#
# 		# Mask Polygon
# 		# Pad to ensure proper polygons for masks that touch image edges.
# 		padded_mask = np.zeros(
# 			(mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
# 		padded_mask[1:-1, 1:-1] = mask
# 		contours = find_contours(padded_mask, 0.5)
# 		for verts in contours:
# 			# Subtract the padding and flip (y, x) to (x, y)
# 			verts = np.fliplr(verts) - 1
# 			p = patches.Polygon(verts, facecolor="none", edgecolor=color, alpha=0.2)
# 			ax.add_patch(p)
# 	ax.imshow(masked_image)#.astype(np.uint8))
# 	if auto_show:
# 		plt.show()

def display_instances(image, boxes, masks, class_ids, class_names,
					  scores=None, title="",
					  figsize=(16, 16), ax=None,
					  show_mask=True, show_bbox=True,
					  colors=None, captions=None):
	"""
	boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
	masks: [height, width, num_instances]
	class_ids: [num_instances]
	class_names: list of class names of the dataset
	scores: (optional) confidence scores for each box
	title: (optional) Figure title
	show_mask, show_bbox: To show masks and bounding boxes or not
	figsize: (optional) the size of the image
	colors: (optional) An array or colors to use with each object
	captions: (optional) A list of strings to use as captions for each object
	"""
	# Number of instances
	N = boxes.shape[0]
	if not N:
		print("\n*** No instances to display *** \n")
	else:
		assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

	# If no axis is passed, create one and automatically call show()
	auto_show = False
	if not ax:
		_, ax = plt.subplots(1, figsize=figsize)
		auto_show = True

	# Generate random colors
	colors = colors or random_colors(N)

	# Show area outside image boundaries.
	#height, width = image.shape[:2]
	#ax.set_ylim(height + 10, -10)
	#ax.set_xlim(-10, width + 10)
	ax.axis('off')
	#ax.set_title(title)

	masked_image = image#image.astype(np.uint32).copy()
	for i in range(N):
		color = colors[i]

		# Bounding box
		if not np.any(boxes[i]):
			# Skip this instance. Has no bbox. Likely lost in image cropping.
			continue
		y1, x1, y2, x2 = boxes[i]
		if show_bbox:
			p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1,
								alpha=0.2, linestyle="dashed",
								edgecolor=color, facecolor='none')
			ax.add_patch(p)

		# Label
		if not captions:
			class_id = class_ids[i]
			score = scores[i] if scores is not None else None
			label = class_names[class_id]
			caption = "{} {:.3f}".format(label, score) if score else label
		else:
			caption = captions[i]
		ax.text(x1, y1, caption,
				color='w', size=5, backgroundcolor="none") # was y1+8

		# Mask
		mask = masks[:, :, i]
		if show_mask:
			masked_image = apply_mask(masked_image, mask, color)

		# Mask Polygon
		# Pad to ensure proper polygons for masks that touch image edges.
		padded_mask = np.zeros(
			(mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
		padded_mask[1:-1, 1:-1] = mask
		contours = find_contours(padded_mask, 0.5)
		for verts in contours:
			# Subtract the padding and flip (y, x) to (x, y)
			verts = np.fliplr(verts) - 1
			p = patches.Polygon(verts, facecolor="none", edgecolor=color, alpha=0.2)
			ax.add_patch(p)
	ax.imshow(masked_image)#.astype(np.uint8))
	if auto_show:
		plt.show()

def display_differences(image,
						gt_box, gt_class_id, gt_mask,
						pred_box, pred_class_id, pred_score, pred_mask,
						class_names, title="", ax=None,
						show_mask=True, show_box=True,
						iou_threshold=0.5, score_threshold=0.5):
	"""Display ground truth and prediction instances on the same image."""
	# Match predictions to ground truth
	gt_match, pred_match, overlaps = utils.compute_matches(
		gt_box, gt_class_id, gt_mask,
		pred_box, pred_class_id, pred_score, pred_mask,
		iou_threshold=iou_threshold, score_threshold=score_threshold)
	# Ground truth = green. Predictions = red
	colors = [(0, 1, 0, .8)] * len(gt_match)\
		   + [(1, 0, 0, 1)] * len(pred_match)
	# Concatenate GT and predictions
	class_ids = np.concatenate([gt_class_id, pred_class_id])
	scores = np.concatenate([np.zeros([len(gt_match)]), pred_score])
	boxes = np.concatenate([gt_box, pred_box])
	masks = np.concatenate([gt_mask, pred_mask], axis=-1)
	# Captions per instance show score/IoU
	captions = ["" for m in gt_match] + ["{:.2f} / {:.2f}".format(
		pred_score[i],
		(overlaps[i, int(pred_match[i])]
			if pred_match[i] > -1 else overlaps[i].max()))
			for i in range(len(pred_match))]
	# Set title if not provided
	title = title or "Ground Truth and Detections\n GT=green, pred=red, captions: score/IoU"
	# Display
	display_instances(
		image,
		boxes, masks, class_ids,
		class_names, scores, ax=ax,
		show_bbox=show_box, show_mask=show_mask,
		colors=colors, captions=captions,
		title=title)

def write_detections_like_cellpose(filepath):
	from PIL import Image
	assert os.path.exists(filepath), "Filepath {} does not exist".format(filepath)
	if filepath[-1]=="/":
		filepath=filepath[:-1]
	assert filepath[filepath.rfind('.')+1:]=="npz", "File must be in .npz format"
	fname =os.path.basename(filepath)
	fname = fname[:fname.find(".")]
	loaded = np.load(filepath,allow_pickle=True)
	masks = loaded['masks']
	out = np.zeros(shape=(masks.shape[0], masks.shape[1],3))
	colors = random_colors(masks.shape[2])
	for sl in range(masks.shape[2]):
		(r,g,b) = colors[sl]
		B = masks[:,:,sl]
		inds = np.argwhere(B)
		rmin = np.min(inds[:,0])
		cmin = np.min(inds[:,1])
		rmax = np.max(inds[:,0])
		cmax = np.max(inds[:,1])
		#draw bounding box
		out[rmin:rmax, cmin, 0]=r
		out[rmin:rmax, cmin, 1]=g
		out[rmin:rmax, cmin, 2]=b
		out[rmin, cmin:cmax, 0]=r
		out[rmin, cmin:cmax, 1]=g
		out[rmin, cmin:cmax, 2]=b
		out[rmin:rmax, cmax, 0]=r
		out[rmin:rmax, cmax, 1]=g
		out[rmin:rmax, cmax, 2]=b
		out[rmax, cmin:cmax, 0]=r
		out[rmax, cmin:cmax, 1]=g
		out[rmax, cmin:cmax, 2]=b
		#draw cell.
		out[:,:,0] = np.where(B>0., r, out[:,:,0])
		out[:,:,1] = np.where(B>0., g, out[:,:,1])
		out[:,:,2] = np.where(B>0., b, out[:,:,2])

	im = Image.fromarray(out.astype(np.uint8))
	im.save(os.path.join(os.path.dirname(filepath),fname+".png"))


#write a function to visualize the training set.
#we want to show the image, the annotations as different colors.
