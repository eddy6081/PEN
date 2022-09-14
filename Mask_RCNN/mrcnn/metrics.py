import numpy as np
import sys
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import remove_small_holes
from scipy.ndimage import find_objects, label
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

"""
Currently, the detect function of Mask-RCNN returns
just a RGB output image with the object detections. This is fine,
except for the case I imagine where there are a ton of cells.
We could instead have it return the "results" dictionary,
which now contains "rois", "class_ids", "scores", "masks".
Run detect with Mask-RCNN and put in a pdb line.
"""

def IoU(M1, M2):
	"""
	Calculate intersection over union between two masked images M1 and M2

	Parameters
	----------------
	M1, M2: boolean, 2D array
			predicted mask
			size [H x W]

	Returns
	---------------
	IoU: float

	"""
	intersection = np.sum(np.logical_and(M1,M2)).astype(np.float)
	union = np.sum(np.logical_or(M1,M2)).astype(np.float)
	return intersection/union, intersection, union

def pred_to_gt_assignment(all_ious, alpha=0.5):
	"""
	Does linear sum assignment strategy based on intersection over union data at
	a given alpha (min IoU) value.

	Parameters
	----------------
	all_ious: list (length K images) of 2D numpy arrays, each with
		shape = [N ground truth objects x M detection objects]
		intersection over union value between nth ground truth object and mth detection object

	alpha: float
		Intersection over union threshold for true-positive and false-positive.

	Returns
	---------------
	all_matches: list of 2D boolean arrays (list length equal to length of alpha)
		boolean True if linear_sum_assignment assigned a prediction to ground
		truth (true-positive), False otherwise.

	"""
	all_matches = [] #useful for making P vs R curve.
	#############################################################################
	for ind,ious in enumerate(all_ious):
		###############################################################################
		#threshold ious with alpha
		ious = np.where(ious>=alpha, ious, 0.)
		#now, we need to assign the ious to their proper labels
		#import pdb;pdb.set_trace()
		n_min = min(ious.shape[0], ious.shape[1])
		# costs = -(ious >= alpha).astype(float) - ious / (2*n_min)
		#threshold ious according to
		costs = -ious
		true_ind, pred_ind = linear_sum_assignment(costs)
		match_ok = ious[true_ind, pred_ind] >= alpha
		# tp = match_ok.sum()
		matched = np.zeros(shape=ious.shape,dtype=np.bool)
		for i,(r,c) in enumerate(zip(true_ind,pred_ind)):
			if match_ok[i]:
				matched[r,c]=True
		all_matches.append(matched)

	# #We can't do a Precision vs Recall curve because of this, it works where you have
	# #bounding boxes predicted with a confidence score, but it doesn't work ehre.
	# #In cellpose documentation, they report the segmentation accuracy, or jcard index.
	return all_matches

#########################################################################################
#########################################################################################
def get_image_regions(pred_masks, gt_masks, size_thresh=100):
	"""
	Parameters
	----------------
	pred_masks: 2D or 3D boolean array
		each item is an array of predicted mask objects
		with size [H x W] or [H x W x N]
	gt_masks: 2D or 3D boolean array
		each item is an array of the ground truth output
		with size [H x W] or [H x W x N] so each channel has only 1 labeled object.
	size_thresh: float
		threshold to apply for smallest object areas.

	Returns
	---------------
	gt_mask_pix: list (length G) of 2D numpy arrays, each with size N x 2
		G number of ground truth objects, each with N x 2 (x,y) pixels assigned to it.

	pred_mask_pix: list (length P) of 2D numpy arrays, each with size M x 2
		P number of predicted objects, each with M x 2 (x,y) pixels assigned to it.
	"""
	#
	if gt_masks.ndim<3:
		gt_masks = np.expand_dims(gt_masks,axis=-1)
	if pred_masks.ndim<3:
		pred_masks = np.expand_dims(pred_masks,axis=-1)

	#gt_mask_pix = np.zeros(gt_masks.shape[2],dtype=np.object) #faster than append method.
	#we want to make the same type as gt_labels though. So a list of arrays.
	gt_mask_pix = []

	for ch in range(gt_masks.shape[2]):
		gt_labels = gt_masks[:,:,ch]
		#the previous steps can make small objects that should not be labeled.
		gt_labels = fill_holes_and_remove_small_masks(gt_labels, min_size=size_thresh, take_max = False)
		#in this case, there should only be 1 object in the channel, filtered by max area.
		#changed CE 06/10/22: make all pixels belong to the one object.
		gt_labels[gt_labels>1]=1
		#now, pull object pixels.
		if len(np.unique(gt_labels))==1 or len(np.unique(gt_labels))>2:
			print("something went wrong")
			import pdb;pdb.set_trace()
		for obj in range(1,len(np.unique(gt_labels))): #in this case, the range should just be 1.
			inds = np.argwhere(gt_labels==obj) #N x 2 array.
			#gt_mask_pix[ch] = inds
			gt_mask_pix.append(inds)

	pred_mask_pix = []

	for ch in range(pred_masks.shape[2]):
		ch_pred_mask = pred_masks[:,:,ch]
		#the previous steps can make small objects that should not be labeled.
		pred_labels = fill_holes_and_remove_small_masks(ch_pred_mask, min_size=size_thresh, take_max = False)
		#changed CE 06/10/22: make all pixels belong to the one object.
		pred_labels[pred_labels>1]=1
		#now pull object pixels.
		for obj in range(1,len(np.unique(pred_labels))):
			inds = np.argwhere(pred_labels==obj)
			pred_mask_pix.append(inds)

	return gt_mask_pix, pred_mask_pix

def get_matched_mask_qualities(all_matches, all_ious):
	"""
	Calculates the mean intersection over union of ONLY predicted masks that
	match with ground truth labels. This reports a measure of how good the
	matched quality of masks are.

	all_matches: list of lists of numpy boolean arrays with shape
				 N ground truth objects x M predicted objects
				 outter list has length equal to number of IoU cutoffs,
				 inner list has length equal to number of analyzed images.

	all_ious: list of lists of numpy float arrays with shape
			  N ground truth objects x M predicted objects
	"""
	Q = np.zeros(len(all_matches))
	for iou_cut_i in range(len(all_matches)):
		matched_ious=[]
		for im_i in range(len(all_matches[iou_cut_i])):
			inds = np.argwhere(all_matches[iou_cut_i][im_i])
			matched_ious.extend(list(all_ious[im_i][inds[:,0],inds[:,1]]))
		Q[iou_cut_i] = np.mean(matched_ious) if len(matched_ious)>0 else 0.
	return Q

def get_cellpose_precision(all_matches):
	"""
	CellPose paper calculates the average precision as:
	"The average precision score is computed from the proportion of matched
	and missed masks."

	INPUTS
	----------------------------------------------------
	all_matches: list of lists of numpy boolean arrays with shape
				 N ground truth objects x M predicted objects
				 outter list has length equal to number of IoU cutoffs,
				 inner list has length equal to number of analyzed images.

	RETURNS
	----------------------------------------------------
	TP, FP, FN, AP
	"""
	#TP = [np.sum([np.sum(im_match,axis=1) for im_match in x]) for x in all_matches]
	TP = [np.sum([np.sum(np.any(im_match,axis=1)) for im_match in x]) for x in all_matches]
	#if any in a row is matched, then we have a true positive.
	FP = [np.sum([np.sum(np.all(~im_match,axis=0)) for im_match in x]) for x in all_matches]
	#if a column has no matches, then it is a false positive
	FN = [np.sum([np.sum(np.all(~im_match,axis=1)) for im_match in x]) for x in all_matches]
	#if a row has no matches, then it is a false negative.
	AP = [tp / (tp + fp + fn) for (tp, fp, fn) in zip(TP, FP, FN)]

	return TP, FP, FN, AP

def calculate_IoUs_v2(gt_mask_pixels, pred_mask_pixels):
	"""
	Calculates intersection over union

	Does matches across prediction channels as well - the models may not put items
	in the correct output channel, but still detects them. I think in this case we
	do not necessarily want to say it missed the detection.  This script does that.

	Parameters
	----------------

	Returns
	---------------
	all_intersections: list of 2D numpy arrays, each with
		shape = [N ground truth objects x M detection objects]
		intersection value between nth ground truth object and mth detection object

	all_unions: list of 2D numpy arrays, each with
		shape = [N ground truth objects x M detection objects]
		union value between nth ground truth object and mth detection object

	all_ious: list of 2D numpy arrays, each with
		shape = [N ground truth objects x M detection objects]
		intersection over union value between nth ground truth object and mth detection object

	"""
	#want to form an output array that is shape = len(gt_mask_pixels) x len(pred_mask_pixels)
	iou_matrix = np.zeros(shape = (len(gt_mask_pixels), len(pred_mask_pixels)))
	intersection_matrix = np.zeros(shape = (len(gt_mask_pixels), len(pred_mask_pixels)))
	union_matrix = np.zeros(shape = (len(gt_mask_pixels), len(pred_mask_pixels)))
	#okay, now for each item in gt_mask_pixels and pred_mask pixels, we want to
	#calculate the intersection and the union.
	for i,gt_obj in enumerate(gt_mask_pixels):
		#gt_obj is an M x 2 array
		for j,pred_obj in enumerate(pred_mask_pixels):
			#pred_obj is an N x 2 array
			#concatenate the arrays.
			together = np.concatenate([gt_obj, pred_obj])
			unq, count = np.unique(together, axis=0, return_counts=True)
			intersection = np.sum(count>1.)
			union = unq.shape[0]
			iou_matrix[i,j] = intersection/union
			intersection_matrix[i,j] = intersection
			union_matrix[i,j] = union
			#intersection is anyrows in gt_obj and pred_obj that are the same.
			#union is the number of unique rows in combined gt_obj and pred_obj.
	return iou_matrix, intersection_matrix, union_matrix

def calculate_average_ious(all_ious, all_matches):
	matched_ious_inds = [np.argwhere(x) for x in all_matches]
	matched_ious = [np.array([all_ious[i][x[0],x[1]] for x in f]) for i,f in enumerate(matched_ious_inds)]
	mean_ious = [np.mean(x) for x in matched_ious]
	return mean_ious

def calculate_average_precision(matches, all_ious, all_intersections, all_unions, all_gt_mask_pix, all_pred_mask_pix):
	"""
	Calculates the per image average
	"""
	#for a given alpha, there is an list of 2d arrays that correspond to each image. This is matches.
	tp_inds = [np.argwhere(x) for x in matches]
	tp_ints = [np.array([ious[x[0],x[1]] for x in tp_inds[i]]) for i,ious in enumerate(all_intersections)]
	tp_unions = [np.array([ious[x[0],x[1]] for x in tp_inds[i]]) for i,ious in enumerate(all_unions)]
	fp_inds = [np.array([col for col in np.argwhere(np.all(~x,axis=0))]) for (x,y) in zip(matches,all_ious)]
	fn_inds = [np.array([row for row in np.argwhere(np.all(~x,axis=1))]) for (x,y) in zip(matches,all_ious)]
	#find the total sum of false positive pixels.
	#all_gt_mask_pix[i] is a list of nx2 array, with length M objects
	n_unmatched_FP_pix = np.array([np.sum([x.shape[0] for j,x in enumerate(all_pred_mask_pix[i]) if j in fp_preds]) for i,fp_preds in enumerate(fp_inds)])
	#above is the number of pixels in each spurious predition object that make up false negatives
	#find the total sum of false negative pixels.
	n_unmatched_FN_pix = np.array([np.sum([x.shape[0] for j,x in enumerate(all_gt_mask_pix[i]) if j in fn_gt]) for i,fn_gt in enumerate(fn_inds)])
	#above is the number of pixels in missed ground truth objects (not matched to any), false negatives

	ttl_tp_int = np.array([np.sum([tp_ints[i]]) for i in range(len(tp_ints))])
	#above is the total number of true positive pixels for detected ground truth.
	ttl_tp_un = np.array([np.sum([tp_unions[i]]) for i in range(len(tp_unions))])
	#above is the number of true positive pixels of detected ground truth + the
	#number of false positive pixels of the predicted shape + the number of
	#false negative pixels that were missed on the ground truth object.
	#import pdb;pdb.set_trace()
	image_AP = np.sum(ttl_tp_int) / (np.sum(ttl_tp_un) + np.sum(n_unmatched_FP_pix) + np.sum(n_unmatched_FN_pix))
	return image_AP

#########################################################################################
#########################################################################################

def plot_PvsR_curve(precision_data, recall_data):
	fig,ax = plt.subplots(1)
	ax.plot(recall_data,precision_data,'r-')
	ax.set_xlabel("Recall")
	ax.set_ylabel("Precision")
	plt.show()

def calculate_mean_average_precision(precision_data, recall_data):
	#already ordered.
	mAP = np.sum([(recall_data[i]-recall_data[i-1])*precision_data[i] for i in range(1,len(recall_data))])
	return mAP


def fill_holes_and_remove_small_masks(masks, min_size=50, take_max=False):
	"""

	fill holes in 2D masks using scipy.ndimage.morphology.binary_fill_holes
	and discard masks smaller than min_size

	Parameters
	----------------
	masks: bool, 2D array
		thresholded masks
		size [H x W]
	min_size: int (optional, default 15)
		minimum number of pixels per mask, can turn off with -1
	take_max: boolean
		set if there should only be 1 object returned. Useful for ground-truth
		mask operation.
	Returns
	---------------
	masks: int, 2D or 3D array
		masks with holes filled and masks smaller than min_size removed,
		0=NO masks; 1,2,...=mask labels,
		size [H x W]

	"""
	connection_array=np.ones((3,3))
	if masks.ndim > 3 or masks.ndim < 2:
		raise ValueError('fill_holes_and_remove_small_masks takes 2D array, not %dD array'%masks.ndim)

	masks = label(masks,connection_array)[0]
	slices = find_objects(masks) #here, expects labels of objects. #returns slices of object
	if not take_max:
		j = 0
		for i,slc in enumerate(slices):
			if slc is not None:
				msk = masks[slc] == (i+1)
				npix = msk.sum()
				if min_size > 0 and npix < min_size:
					masks[slc][msk] = 0
				elif npix > 0:
					msk = binary_fill_holes(msk)
					masks[slc][msk] = (j+1)
					j+=1
	else:
		#find areas.
		max_area = 0.
		for i,slc in enumerate(slices):
			if slc is not None:
				msk = masks[slc] == (i+1)
				npix = msk.sum()
				if npix>max_area:
					max_area = npix
		#take only one object.
		for i,slc in enumerate(slices):
			if slc is not None:
				msk = masks[slc] == (i+1)
				npix = msk.sum()
				if npix==max_area:
					msk = binary_fill_holes(msk)
					masks[slc][msk] = 1
				else:
					masks[slc][msk] = 0

	return masks

def progbar(curr, total, full_progbar):
	frac = curr/total
	filled_progbar = round(frac*full_progbar)
	print('\r', '#'*filled_progbar + '-'*(full_progbar-filled_progbar), '[{:>7.2%}]'.format(frac), end='')
	sys.stdout.flush()
