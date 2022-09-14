"""
From CellPose
"""
import time
from scipy.ndimage.filters import maximum_filter1d
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage import label
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
# def steps3D(p, dP, inds, niter):
# 	""" run dynamics of pixels to recover masks in 3D
#
# 	Euler integration of dynamics dP for niter steps
# 	Parameters
# 	----------------
# 	p: float32, 4D array
# 		pixel locations [axis x Lz x Ly x Lx] (start at initial meshgrid)
# 	dP: float32, 4D array
# 		flows [axis x Lz x Ly x Lx]
# 	inds: int32, 2D array
# 		non-zero pixels to run dynamics on [npixels x 3]
# 	niter: int32
# 		number of iterations of dynamics to run
# 	Returns
# 	---------------
# 	p: float32, 4D array
# 		final locations of each pixel after dynamics
# 	"""
# 	shape = p.shape[1:]
# 	for t in range(niter):
# 		#pi = p.astype(np.int32)
# 		for j in range(inds.shape[0]):
# 			z = inds[j,0]
# 			y = inds[j,1]
# 			x = inds[j,2]
# 			p0, p1, p2 = int(p[0,z,y,x]), int(p[1,z,y,x]), int(p[2,z,y,x])
# 			p[0,z,y,x] = min(shape[0]-1, max(0, p[0,z,y,x] - dP[0,p0,p1,p2]))
# 			p[1,z,y,x] = min(shape[1]-1, max(0, p[1,z,y,x] - dP[1,p0,p1,p2]))
# 			p[2,z,y,x] = min(shape[2]-1, max(0, p[2,z,y,x] - dP[2,p0,p1,p2]))
# 	return p
#
# def steps2D(p, dP, inds, niter):
# 	""" run dynamics of pixels to recover masks in 2D
#
# 	Euler integration of dynamics dP for niter steps
# 	Parameters
# 	----------------
# 	p: float32, 3D array
# 		pixel locations [axis x Ly x Lx] (start at initial meshgrid)
# 	dP: float32, 3D array
# 		flows [axis x Ly x Lx]
# 		[Masks, Xgrad, Ygrad] x Ly x Lx
# 	inds: int32, 2D array
# 		non-zero pixels to run dynamics on [npixels x 2]
# 	niter: int32
# 		number of iterations of dynamics to run
# 	Returns
# 	---------------
# 	p: float32, 3D array
# 		final locations of each pixel after dynamics
# 	"""
# 	shape = p.shape[1:]
# 	tic = time.time()
# 	for t in range(niter):
# 		print(t)
# 		#pi = p.astype(np.int32)
# 		if t==1:
# 			toc = time.time()-tic
# 			print("time to complete 1 iteration of {} took {} sec.".format(niter,toc))
# 			print("estimated time remaining = {} sec.".format(toc*niter))
# 		for j in range(inds.shape[0]):
# 			y = inds[j,0]
# 			x = inds[j,1]
# 			p0, p1 = int(p[0,y,x]), int(p[1,y,x])
# 			p[0,y,x] = min(shape[0]-1, max(0, p[0,y,x] - dP[1,p0,p1]))
# 			#the min keeps pixel location less than the size of the image.
# 			p[1,y,x] = min(shape[1]-1, max(0, p[1,y,x] - dP[2,p0,p1]))
# 	return p

def steps2D_fast(dP, inds, niter, add_noise=False, verbose=True):
	""" run dynamics of pixels to recover masks in 2D
	CE: About 10-50x faster than original CellPose code.

	Euler integration of dynamics dP for niter steps
	Parameters
	----------------
	p: float32, 3D array
		pixel locations [axis x Ly x Lx] (start at initial meshgrid)
	dP: float32, 3D array
		flows [axis x Ly x Lx]
		[Masks, Xgrad, Ygrad] x Ly x Lx
	inds: int32, 2D array
		non-zero pixels to run dynamics on [npixels x 2]
	niter: int32
		number of iterations of dynamics to run
	Returns
	---------------
	Po: float32, 3D array
		final locations of each pixel after dynamics
	"""
	#inds is x y coords.
	#INDS GOES LIKE Y COORDINATE, THEN X COORDINATE.
	Po = inds.copy().astype('float32')
	shape = np.expand_dims(np.array(dP.shape[1:],dtype='float32'),axis=0) #1,2 array.
	#p is an nx2 array
	tic = time.time()
	for n in range(niter):
		if verbose:
			if n==1:
				toc = time.time()-tic
				print("time to complete 1 iteration of {} took {} sec to find centers.".format(niter,toc))
				final_est=toc*niter
				print("estimated time remaining = {} sec for {} iterations.".format(toc*niter, niter-n))
			if n%100==0 and n>0:
				tnow = toc*(niter-n)
				print("estimated time remaining = {} sec for {} iterations.".format(tnow, niter-n))
		#could add thermal noise. %plus or minus one. %add / subtract 1?
		#randomly add 0, -1, or 1.
		if add_noise:
			if n%50==0 and n>100 and n!=niter-1 and n<300:
				F = np.random.uniform(size=Po.shape)
				G = np.random.uniform(size=Po.shape)
				G[F<0.33]=0
				G[F>=0.33]=1
				G[F<0.66]*=-1
				Po+=G
		#import pdb;pdb.set_trace()
		#we need some kind of test for this.
		if n>=niter-100:
			Po += np.flip([3*dP[1:-1,int(x[0]),int(x[1])] for x in np.round(Po)],axis=1)
			Po = np.where(Po>shape-1,shape-1,Po) #sets max appropriately. checked CE.
			Po = np.where(Po<0, 0, Po) #sets minimums appropriately.
		else:
			Po += np.flip([dP[1:-1,int(x[0]),int(x[1])] for x in np.round(Po)],axis=1)
			Po = np.where(Po>shape-1,shape-1,Po) #sets max appropriately. checked CE.
			Po = np.where(Po<0, 0, Po) #sets minimums appropriately.
	return Po

def follow_flows(dP, niter=200, mask_threshold=0.5, border_threshold=0.8, use_gpu=False, verbose=True):
	""" define pixels and run dynamics to recover masks in 2D

	Pixels are meshgrid. Only pixels with non-zero cell-probability
	are used (as defined by inds)
	Parameters
	----------------
	dP: float32, 3D or 4D array
		flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]
		[Mask, combined gradients]
	niter: int (optional, default 200)
		number of iterations of dynamics to run
	interp: bool (optional, default True)
		interpolate during 2D dynamics (not available in 3D)
		(in previous versions + paper it was False)
	use_gpu: bool (optional, default False)
		use GPU to run interpolated dynamics (faster than CPU)
	Returns
	---------------
	p: float32, 3D array
		final locations of each pixel after dynamics
	"""
	shape = np.array(dP.shape[1:]).astype(np.int32)
	niter = np.int32(niter)
	if len(shape)>2:
		p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
				np.arange(shape[2]), indexing='ij')
		p = np.array(p).astype(np.float32)
		# run dynamics on subset of pixels
		#inds = np.array(np.nonzero(dP[0]!=0)).astype(np.int32).T
		inds = np.array(np.nonzero(np.abs(dP[0])>1e-3)).astype(np.int32).T
		p = steps3D(p, dP, inds, niter)
	else:
		p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
		p = np.array(p).astype(np.float32)
		# run dynamics on subset of pixels, those out of the Mask.
		#inds = np.array(np.nonzero(dP[0]>1e-2)).astype(np.int32).T
		#unfortunately, the gradient doesn't always have good numbers as it does on the mask.
		#try running on gradient.
		#inds = np.array(np.nonzero(np.abs(dP[1])>7e-1)).astype(np.int32).T
		#speed this up by eliminating all points not contained in a mask.
		#import pdb;pdb.set_trace()
		#import pdb;pdb.set_trace()
		#inds = np.array(np.nonzero(np.max((np.abs(np.where(dP[0]>mask_threshold, dP[1], 0.)),np.abs(np.where(dP[0]>mask_threshold, dP[2], 0.))))>gradient_threshold)).astype(np.int32).T
		#inds = np.array(np.nonzero(np.where(dP[0]>threshold, dP[1], 0.))>5e-1).astype(np.int32).T
		#inds = np.array(np.nonzero(dP[0]>mask_threshold)).astype(np.int32).T
		#Previously we are getting stuck on some extensions of the cells.
		#New idea, subtract the border from the mask prediction.
		inds = np.array(np.nonzero(np.logical_and(dP[0]>=mask_threshold,np.logical_not(dP[3]>=border_threshold)))).astype(np.int32).T
		if len(inds)>0:
			p = steps2D_fast(dP, inds, niter, verbose=verbose)
		else:
			if verbose:
				print("No cell pixels detected!")
			p = []
	return p

def remove_bad_flow_masks(masks, flows, threshold=0.4):
	""" remove masks which have inconsistent flows

	Uses metrics.flow_error to compute flows from predicted masks
	and compare flows to predicted flows from network. Discards
	masks with flow errors greater than the threshold.
	Parameters
	----------------
	masks: int, 2D or 3D array
		labelled masks, 0=NO masks; 1,2,...=mask labels,
		size [Ly x Lx] or [Lz x Ly x Lx]
	flows: float, 3D or 4D array
		flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]
	threshold: float (optional, default 0.4)
		masks with flow error greater than threshold are discarded.
	Returns
	---------------
	masks: int, 2D or 3D array
		masks with inconsistent flow masks removed,
		0=NO masks; 1,2,...=mask labels,
		size [Ly x Lx] or [Lz x Ly x Lx]

	"""
	merrors, _ = flow_error(masks, flows)
	badi = 1+(merrors>threshold).nonzero()[0]
	masks[np.isin(masks, badi)] = 0
	return masks

def get_masks(centers, flows, rpad=10, edge_d = 5, threshold=0.4, niter=300, size_threshold=30, verbose=True):
	"""
	Inputs
	----------------
		centers: N x D array (N number of points tracked [output of follow_flows])
		flows: 3 x Ly x Lx array
			   flows[0] is output sigmoid activation mask (0-1 range)
			   flows[1] is X-gradient estimation
			   flows[2] is Y-gradient estimation
		rpad: integer padding for histogram binning purposes
		edge_d: integer input for width of histogram bins
		threshold: binary threshold for mask input
		niter: number of iterations to track position of cell-pixels during flow

	Outputs
	----------------
		seeds: M x 2 array, containing x-y coordinates of estimated cell centers
		labels: Ly x Lx array, containing the labeled pixels of each cell
	"""
	#for histogramdd, each row is a coordinate in a D-dimensional space (so, NxD array)
	#centers right now is Nx2 (x,y,z)
	labels=np.zeros_like(flows[0])
	edges=[]
	dims = centers.shape[1]
	###########Start by finding centers##########
	shape0=flows[0].shape
	shape = np.expand_dims(np.array(flows[0].shape),axis=0) #1,2 array.
	for d in range(len(shape0)):
		edges.append(np.arange(-.5-rpad, shape0[d]+.5+rpad, edge_d))
	h,_ = np.histogramdd(centers,bins=edges)
	hmax = h.copy()
	for i in range(dims):
		hmax = maximum_filter1d(hmax, 5, axis=i)
	seeds = np.nonzero(np.logical_and(h-hmax>-1e-6, h>50)) ###NOTE: THIS THRESHOLD SHOULD CHANGE FOR SMALLER CELLS!
	#center locations in normal size image
	seeds = np.stack([x*edge_d - rpad for x in seeds],axis=1) #does both dimensions.
	#if plotting, goes as x=seeds[0][0], y=seeds[0][1]
	#at this point, all cells will be labeled based off of index in seeds.
	#import pdb;pdb.set_trace()

	#if len(seeds)>1:
	if seeds.shape[0]>0:
		this_mask = flows[0]>=threshold
		#this_mask = binary_fill_holes(this_mask)
		#should remove small masks to speed things up
		labeled_mask = label(this_mask)
		for i in range(1,labeled_mask[1]+1):
			if np.sum(labeled_mask[0]==i)<size_threshold: # SIZE THRESHOLD!!!!
				this_mask = np.where(labeled_mask[0]==i,False,this_mask)
		##########flow points in mask for n iterations############
		yy,xx=np.nonzero(this_mask) #indices where mask is nonzero.
		if len(yy)>0: #check that all objects were not removed due to size threshold.
			Po = np.stack((yy,xx),axis=1).astype('float64') #end points
			init_pos=Po.copy() #initial positions
			#in correct form to call Xgrad[Po[0][0], Po[0][1]]
			#for all these points. flow for n iterations.
			#to speed up operation, do mulitply here.
			fastflows = 2*flows
			tic = time.time()
			for n in range(niter):
				if verbose:
					if n==1:
						toc = time.time()-tic
						print("time to complete 1 iteration of {} took {} sec to generate mask.".format(niter,toc))
						final_est=toc*niter
						print("estimated time remaining = {} sec for {} iterations.".format(toc*niter, niter-n))
					if n%100==0 and n>0:
						tnow = toc*(niter-n)
						print("estimated time remaining = {} sec for {} iterations.".format(tnow, niter-n))
				if n>=niter-100:
					Po += np.flip([fastflows[1:-1,int(x[0]),int(x[1])] for x in np.round(Po)],axis=1)
					Po = np.where(Po>shape-1,shape-1,Po) #sets max appropriately. checked CE.
					Po = np.where(Po<0, 0, Po) #sets minimums appropriately.
				else:
					Po += np.flip([flows[1:-1,int(x[0]),int(x[1])] for x in np.round(Po)],axis=1)
					Po = np.where(Po>shape-1,shape-1,Po) #sets max appropriately. checked CE.
					Po = np.where(Po<0, 0, Po) #sets minimums appropriately.
			###############Label the ground truth mask###############
			labeled_mask = label(binary_fill_holes(this_mask), np.ones((3,3)))[0] #makes 8 bit connected
			##########Find closest distance to seed centers##########
			if verbose:
				print("finding closest seed...")
			#import pdb;pdb.set_trace()
			closest=np.array([np.argmin(np.sum(np.power(seeds - x,2),axis=1)) for x in Po])
			closest += 1 #background is zero. Note, the array conversion. W/o, error since int object is not iterable.
			##########label##########
			if verbose:
				print("filling in labels...")
			# dlab = np.zeros(shape=shape0,dtype=np.bool)#flows[0]>threshold
			# #set seed points to backround zero
			# for [sx,sy] in seeds:
			# 	dlab[int(np.clip(np.round(sx),0,shape0[0]-1)),int(np.clip(np.round(sy),0,shape0[1]-1))] = True
			# import pdb;pdb.set_trace()
			# dlab = distance_transform_edt(~dlab)
			# import pdb;pdb.set_trace()
			# dlab = np.where(flows[0]>threshold,dlab,0.)
			for i,lab in enumerate(closest):
				#require that if the flowed point is not within 15 pixels of the center
				#then it should be labeled as background. Otherwise, add the label.
				# To be clear, this is assuming the flows have managed to bring the
				# point within 15 pixels of the center.
				# This is a problem; we want it to assign it to the closest,
				# connected center, right?

				# if np.sqrt(np.sum(np.power(seeds[lab-1,:]-Po[i,:],2)))<15: #distance threshold
				# 	labels[int(init_pos[i,0]),int(init_pos[i,1])]=lab
				#import pdb;pdb.set_trace()
				close_seed_inds = np.round(seeds[lab-1,:]).astype(np.int)
				pixel_inds = np.round(Po[i,:]).astype(np.int)
				#import pdb;pdb.set_trace()
				if labeled_mask[close_seed_inds[0],close_seed_inds[1]]!=0:
					#this doesn't always work since the mask may have a hole and the seed just so happens
					if labeled_mask[close_seed_inds[0],close_seed_inds[1]] == labeled_mask[pixel_inds[0],pixel_inds[1]]:
						#if the mask labels match, then its probably the one.
						labels[int(init_pos[i,0]),int(init_pos[i,1])]=lab
				else:
					#THIS CAN LEAD TO NOISE DETECTIONS,
					if np.sqrt(np.sum(np.power(seeds[lab-1,:]-Po[i,:],2)))<15: #distance threshold
						labels[int(init_pos[i,0]),int(init_pos[i,1])]=lab

		else:
			if verbose:
				print("No masked objects; removed objects are too small for size threshold = {}.".format(size_threshold))
	else:
		if verbose:
			print("No center seeds were able to be determined.")
	return seeds, labels

# def get_masks(p, iscell=None, rpad=20, flows=None, threshold=0.4):
# 	""" create masks using pixel convergence after running dynamics
#
# 	Makes a histogram of final pixel locations p, initializes masks
# 	at peaks of histogram and extends the masks from the peaks so that
# 	they include all pixels with more than 2 final pixels p. Discards
# 	masks with flow errors greater than the threshold.
# 	Parameters
# 	----------------
# 	p: float32, 3D or 4D array
# 		final locations of each pixel after dynamics,
# 		size [axis x Ly x Lx] or [axis x Lz x Ly x Lx].
# 	iscell: bool, 2D or 3D array
# 		if iscell is not None, set pixels that are
# 		iscell False to stay in their original location.
# 	rpad: int (optional, default 20)
# 		histogram edge padding
# 	threshold: float (optional, default 0.4)
# 		masks with flow error greater than threshold are discarded
# 		(if flows is not None)
# 	flows: float, 3D or 4D array (optional, default None)
# 		flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]. If flows
# 		is not None, then masks with inconsistent flows are removed using
# 		`remove_bad_flow_masks`.
# 	Returns
# 	---------------
# 	M0: int, 2D or 3D array
# 		masks with inconsistent flow masks removed,
# 		0=NO masks; 1,2,...=mask labels,
# 		size [Ly x Lx] or [Lz x Ly x Lx]
#
# 	"""
#
# 	pflows = []
# 	edges = []
# 	shape0 = p.shape[1:]
# 	dims = len(p)
# 	if iscell is not None:
# 		if dims==3:
# 			inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]),
# 				np.arange(shape0[2]), indexing='ij')
# 		elif dims==2:
# 			inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]),
# 					 indexing='ij')
# 		for i in range(dims):
# 			p[i, ~iscell] = inds[i][~iscell]
#
# 	for i in range(dims):
# 		pflows.append(p[i].flatten().astype('int32'))
# 		edges.append(np.arange(-.5-rpad, shape0[i]+.5+rpad, 1))
#
# 	h,_ = np.histogramdd(tuple(pflows), bins=edges)
# 	hmax = h.copy()
# 	for i in range(dims):
# 		hmax = maximum_filter1d(hmax, 5, axis=i)
#
# 	seeds = np.nonzero(np.logical_and(h-hmax>-1e-6, h>10))
# 	Nmax = h[seeds]
# 	isort = np.argsort(Nmax)[::-1]
# 	for s in seeds:
# 		s = s[isort]
#
# 	pix = list(np.array(seeds).T)
#
# 	shape = h.shape
# 	if dims==3:
# 		expand = np.nonzero(np.ones((3,3,3)))
# 	else:
# 		expand = np.nonzero(np.ones((3,3)))
# 	for e in expand:
# 		e = np.expand_dims(e,1)
#
# 	for iter in range(5):
# 		for k in range(len(pix)):
# 			if iter==0:
# 				pix[k] = list(pix[k])
# 			newpix = []
# 			iin = []
# 			for i,e in enumerate(expand):
# 				epix = e[:,np.newaxis] + np.expand_dims(pix[k][i], 0) - 1
# 				epix = epix.flatten()
# 				iin.append(np.logical_and(epix>=0, epix<shape[i]))
# 				newpix.append(epix)
# 			iin = np.all(tuple(iin), axis=0)
# 			for p in newpix:
# 				p = p[iin]
# 			newpix = tuple(newpix)
# 			igood = h[newpix]>2
# 			for i in range(dims):
# 				pix[k][i] = newpix[i][igood]
# 			if iter==4:
# 				pix[k] = tuple(pix[k])
#
# 	M = np.zeros(h.shape, np.int32)
# 	for k in range(len(pix)):
# 		M[pix[k]] = 1+k
#
# 	for i in range(dims):
# 		pflows[i] = pflows[i] + rpad
# 	M0 = M[tuple(pflows)]
#
# 	# remove big masks
# 	_,counts = np.unique(M0, return_counts=True)
# 	big = np.prod(shape0) * 0.4
# 	for i in np.nonzero(counts > big)[0]:
# 		M0[M0==i] = 0
# 	_,M0 = np.unique(M0, return_inverse=True)
# 	M0 = np.reshape(M0, shape0)
#
# 	if threshold is not None and threshold > 0 and flows is not None:
# 		M0 = remove_bad_flow_masks(M0, flows, threshold=threshold)
# 		_,M0 = np.unique(M0, return_inverse=True)
# 		M0 = np.reshape(M0, shape0).astype(np.int32)
#
# 	return M0

def flow_error(maski, dP_net):
	""" error in flows from predicted masks vs flows predicted by network run on image
	This function serves to benchmark the quality of masks, it works as follows
	1. The predicted masks are used to create a flow diagram
	2. The mask-flows are compared to the flows that the network predicted
	If there is a discrepancy between the flows, it suggests that the mask is incorrect.
	Masks with flow_errors greater than 0.4 are discarded by default. Setting can be
	changed in Cellpose.eval or CellposeModel.eval.
	Parameters
	------------

	maski: ND-array (int)
		masks produced from running dynamics on dP_net,
		where 0=NO masks; 1,2... are mask labels
	dP_net: ND-array (float)
		ND flows where dP_net.shape[1:] = maski.shape
	Returns
	------------
	flow_errors: float array with length maski.max()
		mean squared error between predicted flows and flows from masks
	dP_masks: ND-array (float)
		ND flows produced from the predicted masks

	"""
	if dP_net.shape[1:] != maski.shape:
		print('ERROR: net flow is not same size as predicted masks')
		return
	maski = np.reshape(np.unique(maski.astype(np.float32), return_inverse=True)[1], maski.shape)
	# flows predicted from estimated masks
	dP_masks,_ = dynamics.masks_to_flows(maski)
	iun = np.unique(maski)[1:]
	flow_errors=np.zeros((len(iun),))
	for i,iu in enumerate(iun):
		ii = maski==iu
		if dP_masks.shape[0]==2:
			flow_errors[i] += ((dP_masks[0][ii] - dP_net[0][ii]/5.)**2
							+ (dP_masks[1][ii] - dP_net[1][ii]/5.)**2).mean()
		else:
			flow_errors[i] += ((dP_masks[0][ii] - dP_net[0][ii]/5.)**2 * 0.5
							+ (dP_masks[1][ii] - dP_net[1][ii]/5.)**2
							+ (dP_masks[2][ii] - dP_net[2][ii]/5.)**2).mean()
	return flow_errors, dP_masks


def _run_gradient_find(centers, niters, bboxes, masks):
	xgrads = np.zeros(shape=masks.shape).astype(np.float32)
	ygrads = np.zeros(shape=masks.shape).astype(np.float32)
	#so, some objects might be overlapping...
	#run heat diffusion on each mask.
	#then Add all together. Those with multiple masks will need to be looked at carefully. I wonder what they will show?
	s_unit = np.ones((3,3),dtype=np.float32)/9.0
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
		for n in range(niters[i]):
			heat_map[centers[i,1] - bboxes[i,1],centers[i,0]-bboxes[i,0]]+=1
			#Much, much faster doing it this way.
			#convolve with s_unit
			heat_map = scipy.signal.convolve2d(heat_map,s_unit,mode='same')
			heat_map = heat_map * m
		#heat_map = np.log(1+((1000/np.min(heat_map[m==True]))*heat_map))
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

def npz_to_mask(filepath, centers_iter=500, masks_iter=300, mask_thresh=0.4, border_thresh=0.8, size_thresh=30, verbose=True):
	"""
	Inputs
	----------------
		filepath:     /path/to/file.npz, provide path to npz file saved from model prediction
				      generated labeled image will be saved in same location

		centers_iter: number of iterations provided to follow_flows.
					  More iterations allows centers to converge to more precise location,
					  but will take longer to complete.

		masks_iter:   number of iterations provided to get_masks.
					  More iterations may yield better labeling, but takes
					  longer to complete.

		mask_thresh:  threshold provided to get_masks.
					  Threshold value to convert sigmoid activated output mask estimate
					  to binary value. Lower threshold means more pixels will be tracked,
					  yielding longer analysis time. Higher threshold may miss some cell pixels.

		size_thresh: threshold for minimum area of objects.

	Outputs
	----------------
		Labeled pixel map PNG file saved under filepath
		npz compressed file to verify output map saved under filepath
	"""
	assert os.path.exists(filepath), "Filepath {} does not exist".format(filepath)
	if filepath[-1]=="/":
		filepath=filepath[:-1]
	assert filepath[filepath.rfind('.')+1:]=="npz", "File must be in .npz format"
	fname = os.path.basename(filepath)
	fname = fname[:fname.find(".")]
	loaded = np.load(filepath, allow_pickle=True)
	#IM = loaded['image']
	M = loaded['mask']
	X = loaded['xgrad']
	Y = loaded['ygrad']
	B = loaded['edges']
	#fill holes on mask.
	#normalize gradients, if necessary.
	if np.max(np.abs(X))>1.:
		X = X / np.max(np.abs(X))
	if np.max(np.abs(Y))>1.:
		Y = Y / np.max(np.abs(Y))
	if len(M.shape)==2:
		M = np.expand_dims(M, axis=-1)
		X = np.expand_dims(X, axis=-1)
		Y = np.expand_dims(Y, axis=-1)
		B = np.expand_dims(B, axis=-1)
	#E = loaded['edges']
	if len(M.shape)==3:
		"""
		New versions of the code allows for an N channel output.
		"""
		S = []
		L = []
		max_lab = 0
		for ch in range(M.shape[2]):
			if np.sum(M[:,:,ch]>=mask_thresh)>0:
				All = [M[:,:,ch],-X[:,:,ch],-Y[:,:,ch], B[:,:,ch]]
				All = np.stack(All)
				print("Running pixel flow on {} channel {}".format(filepath, ch))
				Points = follow_flows(All, mask_threshold=mask_thresh, border_threshold = border_thresh, niter=centers_iter, verbose=verbose)
				if len(Points)>0:
					sd, lab = get_masks(Points, All, threshold=mask_thresh, niter=masks_iter, size_threshold = size_thresh, verbose=verbose)
					#L has shape like HxW. S has shape N x 2 where N is the number of objects in that image.
					S.append(sd)
					ml = np.max(lab)
					lab = np.where(lab>0., lab+max_lab, 0.)
					L.append(lab)
					max_lab += ml
		#import pdb;pdb.set_trace()
		#Comment next line since each entry has unequal number of detections.
		#S = np.stack(S, axis = -1)
		if len(L)>0:
			L = np.stack(L, axis = -1)

		if verbose:
			print("\nSaving npz compressed file with seeds and labels.")
		np.savez_compressed(os.path.join(os.path.dirname(filepath),fname+"_results"), seeds=S, label=L)

		# if max_lab<=255.0*3:
		# 	if verbose:
		# 		print("Saving labels as png file {}...".format(fname+".png"))
		# 	out = np.zeros(shape=(L.shape[0],L.shape[1],3))
		# 	for obj in range(1,int(max_lab)):
		# 		ch=0
		# 		placed = False
		# 		while ch<3 and not placed:
		# 			#need x y pixels of obj, not sure what channel they are in.
		# 			B = np.max(np.where(L==obj, L, 0),axis=2)
		# 			#Now, need to see if it can fit into the channel.
		# 			if np.sum(np.where(B>0.,out[:,:,ch],0.))==0.0:
		# 				out[:,:,ch] = np.where(B>0., B, out[:,:,ch])
		# 				placed = True
		# 			ch+=1
		#
		# 		if not placed:
		# 			print("Cannot fit object {} of {} into RGB output channels!".format(obj, max_lab-1))
		# 			import pdb;pdb.set_trace()
		#
		# 	im = Image.fromarray(out.astype(np.uint8))
		# 	im.save(os.path.join(os.path.dirname(filepath),fname+".png"))

		#here, you should do bounding box and coloration of cells randomly.
		out = np.zeros(shape=(L.shape[0],L.shape[1],3))
		for _,obj in enumerate(np.unique(L)): #range(1,int(max_lab)):
			if obj>0.0:
				r,g,b = random_bright_color()
				B = np.max(np.where(L==obj, L, 0),axis=2)
				inds = np.argwhere(B) # shape N x 2
				#import pdb;pdb.set_trace()
				rmin = np.min(inds[:,0])
				cmin = np.min(inds[:,1])
				rmax = np.max(inds[:,0])
				cmax = np.max(inds[:,1])
				#draw bounding box.
				out[rmin:rmax, cmin,0]=r
				out[rmin:rmax, cmin,1]=g
				out[rmin:rmax, cmin,2]=b
				out[rmin, cmin:cmax, 0]=r
				out[rmin, cmin:cmax, 1]=g
				out[rmin, cmin:cmax, 2]=b
				out[rmin:rmax, cmax,0]=r
				out[rmin:rmax, cmax,1]=g
				out[rmin:rmax, cmax,2]=b
				out[rmax, cmin:cmax, 0]=r
				out[rmax, cmin:cmax, 1]=g
				out[rmax, cmin:cmax, 2]=b
				#draw cell.
				out[:,:,0] = np.where(B>0., r, out[:,:,0])
				out[:,:,1] = np.where(B>0., g, out[:,:,1])
				out[:,:,2] = np.where(B>0., b, out[:,:,2])

		if verbose:
			print("Saving labels as png file {}...".format(fname+".png"))
		im = Image.fromarray(out.astype(np.uint8))
		im.save(os.path.join(os.path.dirname(filepath),fname+".png"))

	else:
		print("Need help. Output Mask has >3 dimensions...Pausing.")
		import pdb;pdb.set_trace()


def write_to_image(image, filepath,filename):
	#import pdb;pdb.set_trace()
	#scale between 0 and 255.
	if np.max(image)<=1:
		image=image*255
	im = Image.fromarray(image.astype(np.uint8))
	im.save(os.path.join(filepath,os.path.splitext(filename)[0]+"_labels.png"))

# #view keys with sorted(loaded.files)
# f_path = "/users/czeddy/documents/workingfolder/deep_learning_test/cellpose/my_code/results/3DCell_Max/Aggregate_test.npz"
# loaded = np.load(f_path)
# #view keys with sorted(loaded.files)
# IM_3d = loaded['image']
# M_3d = loaded['mask']
# X_3d = loaded['xgrad']
# Y_3d = loaded['ygrad']
# All=[M_3d,-X_3d,-Y_3d]
# All=np.stack(All)
# p = follow_flows(All, niter=500)
# plt.imshow(IM_3d)
# plt.plot(p[:,1],p[:,0],'rx')
# plt.show()
#
# S,L=get_masks(p, All)
# plt.imshow(IM_3d)
# plt.plot(S[:,1],S[:,0],'rx')

def random_bright_color():
	import random
	import colorsys
	h, s, l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
	r, g, b = [int(256*i) for i in colorsys.hls_to_rgb(h,l,s)]
	return r,g,b
