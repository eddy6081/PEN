"""
Author: Christopher Z. Eddy
Date: 06/23/21
Contact: eddych@oregonstate.edu

Object oriented implementation in Tensorflow/Keras
Originally by MouseLand (https://github.com/MouseLand/cellpose) in PyTorch

Purpose is really so we can run an implementation of CellPose just for inference
or to examine the dataset, without any further difficulty, and be able to change
directories on the fly, load different models, all without restarting.

For training, see cell.py.
"""
import argparse
import os
import sys
#import python files
import model as modellib
import config as configurations
import utils
import metrics
import copy
import datetime
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback, TerminateOnNaN
import flows_to_masks
import matplotlib.pyplot as plt

################################################################
################################################################
################CALL BACKS for training ########################
################################################################
################################################################

class MyCallback(Callback):
	def on_epoch_end(self, epoch, logs=None):
		lr = self.model.optimizer.lr
		decay = self.model.optimizer.decay
		iterations = self.model.optimizer.iterations
		lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
		print("Learning Rate = ", K.eval(lr_with_decay))
	def on_train_end(self, logs=None):
		lr = self.model.optimizer.lr
		decay = self.model.optimizer.decay
		iterations = self.model.optimizer.iterations
		lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
		print("Learning Rate = ", K.eval(lr_with_decay))

	def on_train_begin(self,logs=None):
		lr = self.model.optimizer.lr
		decay = self.model.optimizer.decay
		iterations = self.model.optimizer.iterations
		lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
		print("Learning Rate = ", K.eval(lr_with_decay))


class CustomLearningRateScheduler(Callback):
	"""Learning rate scheduler which sets the learning rate according to schedule.

  Arguments:
	  schedule: a function that takes an epoch index
		  (integer, indexed from 0) and current learning rate
		  as inputs and returns a new learning rate as output (float).
  """

	def __init__(self, schedule):
		super(CustomLearningRateScheduler, self).__init__()
		self.schedule = schedule

	def on_epoch_begin(self, epoch, logs=None):
		if not hasattr(self.model.optimizer, "lr"):
			raise ValueError('Optimizer must have a "lr" attribute.')
		# Get the current learning rate from model's optimizer.
		lr = float(K.get_value(self.model.optimizer.learning_rate))
		# Call schedule function to get the scheduled learning rate.
		scheduled_lr = self.schedule(epoch, lr)
		# Set the value back to the optimizer before this epoch starts
		K.set_value(self.model.optimizer.lr, scheduled_lr)
		print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))


LR_SCHEDULE = [
	# (epoch to start, learning rate) tuples
	(3, 0.05),
	(6, 0.01),
	(9, 0.005),
	(12, 0.001),
]

def lr_schedule(epoch, lr):
	"""Helper function to retrieve the scheduled learning rate based on epoch."""
	if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
		return lr
	for i in range(len(LR_SCHEDULE)):
		if epoch == LR_SCHEDULE[i][0]:
			return LR_SCHEDULE[i][1]
	return lr


class RegLossCallback(Callback):
	"""
	Use to verify that the regulation loss decreases. Each weight should have Regularization
	implemented and therefore this should also decrease.
	See https://stackoverflow.com/questions/53715409/manually-computed-validation-loss-different-from-reported-val-loss-when-using-re
	"""
	def __init__(self):#, lambd):
		super(RegLossCallback, self).__init__()
		#self.validation_x = validation_x
		#self.validation_y = validation_y
		#self.lambd = self.model.optimizer.lr#lambd

	def on_epoch_end(self, epoch, logs=None):
		#validation_y_predicted = self.model.predict(self.validation_x)

		# Compute regularization term for each layer
		weights = self.model.trainable_weights
		reg_term = 0
		for i, w in enumerate(weights):
			if i % 2 == 0:  # weights from layer i // 2
				w_f = K.flatten(w)
				#reg_term += self.lambd[i // 2] * K.sum(K.square(w_f))
				reg_term += self.model.optimizer.lr * K.sum(K.square(w_f))

		#mse_loss = K.mean(mean_squared_error(self.validation_y, validation_y_predicted))
		#loss = mse_loss + K.cast(reg_term, 'float64')
		loss = K.cast(reg_term,'float64')

		print("Reg Loss: %.4f" % K.eval(loss))
#ValidationCallback(x_validation, y_validation, lambd)



###########################################################################
###########################################################################
#####################  CELL POSE IMPLEMENTATION  ##########################
###########################################################################
###########################################################################
###########################################################################

class CellPose(object):
	"""
	object oriented CellPose implementation.
	Recommended for inference and any post-processing. Training should work, but
	I prefer to run sh scripts using argparse rather than OOP based code.

	Minimum Working Example (MWE):


	#####################################################################
	Training:

	I do not recommend using this script for ACTUAL training. Please use cell.py
	However, visualizing the model or training/val set is a useful tool.

	CP = CellPose(mode="training", dataset_path="/your/path/to/image_directory",
			data_type="Cell3D", weights="Best_Models/3D/Graph_Max/cellpose20210616T2009/cellpose_0036.h5")
	#you do not have to give it weights.

	CP.create_model()

	CP.model.print_model() #this will save a .png file with model architecture in your cwd.


	#To visualize a training set image, try:

	CP.import_train_val_data()

	CP.see_train_example(0)


	Use CP.config.display() to see your current configuration for training.
	You may change the types of augmentations being used (I have not yet built
	a blur augmentation, so setting that to True will do nothing) or whatever
	else.


	######################################################################
	Inference:

	CP = CellPose(mode="inference", dataset_path="/your/path/to/image_directory",
			data_type="Cell3D", weights="Best_Models/3D/Graph_Max/cellpose20210616T2009/cellpose_0036.h5")

	CP.create_model()

	CP.import_inference_dataset()

	CP.run_inference()

	######################################################################
	Metrics:
	To acquire metrics (mean average precision, precision, recall, jaccard index)

	CP = CellPose(mode="inference", dataset_path="/your/path/to/image_directory",
			data_type="Cell3D", weights="")

	CP.import_inference_dataset()

	results = CP.measure_metrics(alpha = 0.5) #to evaluate at a minimum IoU of 0.5
	OR
	results = CP.measure_metrics(alpha=list(np.linspace(0.5,0.95, int((0.95-0.5)/0.05)+2)))
	OR
	results = CP.measure_metrics(alpha=list(np.linspace(0.5,0.95, int((0.95-0.5)/0.05)+2)), detection_dir = "path/to/saved/.npz files")

	np.savez_compressed("CellPose_metrics.npz", **results)

	######################################################################
	Visualize PEN Output:

	#several things can be done here.
	(1) visualize output from PEN module, if 3D model is used.
	CP = CellPose(mode="visualize", dataset_path="/your/path/to/image_directory",
			data_type="Cell3D", weights="Best_Models/3D/Graph_Max/cellpose20210616T2009/cellpose_0036.h5")
	#must load weights.

	CP.import_inference_dataset()

	CP.create_stem_model()

	CP.run_stem_visualize()

	#######################################################################
	Visualize Effects of Padding on PEN Output:
	This is best to run on GPU, and visualize later on CPU. Save output as npz file.

	Create the CellPose class, load weights.
	CP = CellPose(mode="visualize", dataset_path="/your/path/to/image_directory",
			data_type="Cell3D", weights="Best_Models/3D/Graph_Max/cellpose20210616T2009/cellpose_0036.h5")
	#MUST load weights.

	Load inference dataset.
	CP.import_inference_dataset()

	Create the model.
	CP.create_stem_model()

	Check effects of padding from all combinations.
	z_frontpad, PEN_outputs = CP.stem_padcheck(image_id=0,max_size=1024) #we should introduce the tiling algorithm (from load_image_inference)

	May wish to save as compressed npz file here.
	np.savez_compressed("./PEN_output",zpad = z_frontpad, out = PEN_outputs)

	Run plotting to determine best padding option.
	zbest = utils.plot_padding(z_frontpad,PEN_outputs)

	Run inference with the best padding strategy.
	CP = CellPose(mode="inference", dataset_path="/your/path/to/image_directory",
			data_type="Cell3D", weights="Best_Models/3D/Graph_Max/cellpose20210616T2009/cellpose_0036.h5")

	CP.create_model()

	CP.import_inference_dataset()

	CP.run_inference(z_begin = zbest)

	#######################################################################
	Visualize Learned Filters:

	(2) visualize filters in the pyramid structure.
	Create the CellPose class, load weights.
	CP = CellPose(mode="visualize", dataset_path="/your/path/to/image_directory",
			data_type="Cell3D", weights="Best_Models/3D/Graph_Max/cellpose20210616T2009/cellpose_0036.h5")
	#must load weights.

	create the model.
	CP.create_stem_model()

	Use:
	 mli = CP.pull_layer_names()
	 ind = CP.find_ind_layer("name of layer", mli) #use CP.model.print_model() to get a graphical version with layer names, if you prefer.
	 w = CP.pull_layer_weights(ind[0],mli)
	 CP.visualize_kernel_2d(w, 0, 0) #the first 0 is the learned filter number. the second is the input channel index. See code.

	######################################################################
	Visualize Output:

	#run process_result to first obtain seeds and cell labels.
	process_result("/wherever/CP/saved/the/.npz file")

	visualize_seeds("/wherever process_result/saved/its/output/.npz file")

	visualize_edges("/wherever/CP/saved/the/.npz file")

	######################################################################

	Good luck!
	Contribute by adding more visualization tools (functions) down at the
	bottom of this script!

	"""
	def __init__(self, mode, dataset_path, data_type, weights=None, logs=None, subset=None):
		"""
		mode:
			Command for CellPose.
			Options: "'training', 'inference', or 'visualize'"

		dataset_path:
			Root directory of the dataset. Training MUST have form:
				dataset_directory
				----/train
				--------/images
				--------/gt
				----/val
				--------/images
				--------/gt
				Inference MUST have form:
				dataset_directory
				----/images

		data_type:
			Specifies which type of configuration and model to load.
			Options: 'MouseVision', 'MouseVisionSplit', 'Cell3D', 'Cell2D'"

		weights:
			Path to weights .h5 file if you wish to load previously trained weights.

		logs:
			Logs and checkpoints directory (default=logs/)

		subset:
			Subset of dataset to run prediction on. Generally, okay to exclude.
			ex. 'test' will only pull data from dataset/test"
		"""

		assert mode in ["training", "inference", "visualize"], "mode argument must be one of 'training' or 'inference' or 'visualize'"
		assert data_type in ["MouseVision", "MouseVisionSplit", "Cell3D", "Cell2D"], "data_type argument must be one of 'MouseVision', 'MouseVisionSplit', 'Cell3D', 'Cell2D'"
		if dataset_path[-1]=="/":
			dataset_path=dataset_path[0:-1]

		if mode=="visualize":
			assert weights, "--weights argument must given (path to weights) for visualization of input."
		# if args.mode == "training":
		# 	assert args.dataset, "Argument --dataset is required for training"
		#elif args.command == "detect":
		#    assert args.subset, "Provide --subset to run prediction on"
		############################################################################
		print("Weights path: ", weights)
		self.weights = weights

		print("Dataset directory: ", dataset_path)
		self.dataset_dir = dataset_path
		if subset:
			print("Subset for inference: ", subset)
			self.subset = subset

		ROOT_DIR = os.path.abspath("./")
		DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

		if logs:
			self.logs = logs
		else:
			self.logs = DEFAULT_LOGS_DIR
		print("Logs directory: ", self.logs)

		print("Data from: ", data_type)
		self.data_type = data_type


		#Create appropriate configuration and model. Does not load dataset.
		if self.data_type=="MouseVision":
			self.config = configurations.Config()
			self.dataset=utils.CellPoseDataset()
		elif self.data_type=="MouseVisionSplit":
			self.config = configurations.Config()
			self.dataset=utils.CellPoseDataset()
		elif self.data_type=="Cell3D":
			self.config = configurations.Cell3DConfig()
			self.dataset=utils.CellDataset()
		elif self.data_type=="Cell2D":
			self.config = configurations.Cell2DConfig()
			self.dataset = utils.CellDataset()

		if self.weights:
			self.config = utils.load_config_file(os.path.join(os.path.dirname(self.weights),"README.txt"), self.config)

		if mode=="inference":
			#we do not want to reshape our images.
			#well, that isn't ENTIRELY true. If the image is too small to be processed,
			#then we do need to pad it.
			self.config.IMAGE_RESIZE_MODE='pad64'
			#'none'

		self.config.display()
		self.mode = mode

	#############################################################################
	############################################################################
	def create_model(self):
		#create model instance.
		#here, the model does not have a pre-specified shape (meaning the expected input
		#is None x None x None x C or None x None x None x Z x 1) making it more flexible
		self.model = modellib.CPnet(self.mode, self.config, self.logs)
		#if given, load weights.
		if self.weights:
			modellib.load_weights(self.model.keras_model, self.weights)

	########################################################################
	########################################################################
	"""
	TRAINING RELATED FUNCTIONS
	"""
	########################################################################
	########################################################################
	def import_train_val_data(self):
		if self.data_type=="MouseVision":
			import textwrap
			#MouseVision is not automatically split into two. In this case, lets set
			#the random seed for reproduceability. Secondly, split the dataset into two.
			dataset_all=copy.deepcopy(self.dataset)
			print("Loading Dataset...")
			dataset_all.load_cell(self.dataset_dir)
			np.random.seed(0)
			num_ex = len(dataset_all.image_info)
			train_pick = int(np.round(0.8*num_ex))
			val_pick = int(num_ex-train_pick)
			S="\nNote: MouseVision Datasets have no train/val split. \n\
			Splitting set automatically with 80% data ({} of {} images) in training\n\
			and 20% data ({} of {}) in validation.\n".format(train_pick,num_ex,val_pick,num_ex)
			for line in S.splitlines():
				print(line.strip())
			#can we split it?
			dataset_train = copy.deepcopy(dataset_all)
			dataset_val = copy.deepcopy(dataset_all)
			train_inds = list(np.random.choice(num_ex,train_pick,replace=False))
			train_inds.sort()
			val_inds = [i for i in range(num_ex) if i not in train_inds]
			dataset_train.image_info = [dataset_train.image_info[i] for i in train_inds]
			dataset_val.image_info = [dataset_val.image_info[i] for i in val_inds]
			#pick indices for
			dataset_train.prepare()
			dataset_val.prepare()
			#import pdb;pdb.set_trace()
		else:
			# Training dataset.
			print("Loading the train set")
			dataset_train=copy.deepcopy(self.dataset)
			dataset_train.load_cell(self.dataset_dir, "train")#, model.config.INPUT_DIM)
			dataset_train.prepare()

			# Validation dataset
			print("Loading the validation set")
			dataset_val=copy.deepcopy(self.dataset)
			dataset_val.load_cell(self.dataset_dir, "val")
			dataset_val.prepare()

		self.dataset_train = dataset_train
		self.dataset_val = dataset_val

	def see_train_example(self, image_id=0, ch=0):
		"""
		image_id = index of image you wish to see.
		ch = channel of mask and gradients to display.
		"""
		print("Showing file ", self.dataset_train.image_info[image_id]['id'])
		#IM, M, X, Y, B = utils.load_image_gt(self.dataset_train, self.config, image_id, augmentation=self.config.Augmentors)
		IM, M, X, Y, B = utils.load_image_gt2(self.dataset_train, self.config, image_id, augmentation=self.config.Augmentors)
		fig,((ax1,ax2,ax3),(ax4,ax5,ax6))=plt.subplots(2,3)
		if len(IM.shape)>3:
			ax1.imshow(np.max(IM,axis=2))
		else:
			ax1.imshow(IM)
		ax1.set_title("Input")
		ax1.axis('off')
		print("Input shape = ", IM.shape)
		print("Input dtype = ", IM.dtype)
		print("Input max/min/mean = %.2f/%.2f/%.6f" % (np.max(IM[:]),np.min(IM[:]),np.mean(IM)))
		if M.shape[2]>3:
			ax2.imshow(M[:,:,ch])
			ax3.imshow(X[:,:,ch])
			ax4.imshow(Y[:,:,ch])
			ax5.imshow(B[:,:,ch])
			ax6.imshow(utils.linearly_color_encode_image(IM))
		else:
			ax2.imshow(M.astype(np.float32))
			ax3.imshow(((X/10.)+1.)/2., cmap='bwr')
			ax4.imshow(((Y/10.)+1.)/2.,cmap='bwr')
			ax5.imshow(B.astype(np.float32))
			if self.config.INPUT_DIM=="3D":
				ax6.imshow(utils.linearly_color_encode_image(IM))
		ax2.set_title("GT Mask")
		ax2.axis('off')
		print("GT Mask shape = ", M.shape)
		print("GT Mask dtype = ", M.dtype)
		ax3.set_title("GT X-grad")
		ax3.axis('off')
		print("GT X-grad shape = ", X.shape)
		print("GT X-gray dtype = ", X.dtype)
		ax4.set_title("GT Y-grad")
		ax4.axis('off')
		print("GT Y-grad shape = ", Y.shape)
		print("GT Y-gray dtype = ", Y.dtype)
		ax5.set_title("GT Borders")
		ax5.axis('off')
		print("GT borders shape = ", B.shape)
		print("GT borders dtype = ", B.dtype)
		ax6.set_title("Linear Depth IM")
		ax6.axis('off')

		plt.show()

		#utils.plt_show_many(M,M.shape[2])
		import pdb;pdb.set_trace()

	def see_val_example(self, image_id=0, ch=0):
		print("Showing file ", self.dataset_val.image_info[image_id]['id'])
		IM, M, X, Y, B = utils.load_image_gt(self.dataset_val, self.config, image_id, augmentation=self.config.Augmentors)
		fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2)
		ax1.imshow(np.max(IM,axis=2))
		ax1.set_title("Input")
		ax1.axis('off')
		print("Input shape = ", IM.shape)
		print("Input dtype = ", IM.dtype)
		ax2.imshow(M[:,:,ch])
		ax2.set_title("GT Mask")
		ax2.axis('off')
		print("GT Mask shape = ", M.shape)
		print("GT Mask dtype = ", M.dtype)
		ax3.imshow(X[:,:,ch])
		ax3.set_title("GT X-grad")
		ax3.axis('off')
		print("GT X-grad shape = ", X.shape)
		print("GT X-gray dtype = ", X.dtype)
		ax4.imshow(Y[:,:,ch])
		ax4.set_title("GT Y-grad")
		ax4.axis('off')
		print("GT Y-grad shape = ", Y.shape)
		print("GT Y-gray dtype = ", Y.dtype)
		plt.show()
		import pdb;pdb.set_trace()

	def see_gt_example(self, image_id=0, ch=0):
		"""
		Use for datasets where you have a labeled ground-truth to compare to.
		Use with load_inference_dataset
		"""
		print("Showing file ", self.dataset_test.image_info[image_id]['id'])
		RMODE = self.config.IMAGE_RESIZE_MODE
		self.config.IMAGE_RESIZE_MODE='none'
		IM, M, X, Y, B = utils.load_image_gt(self.dataset_test, self.config, image_id, augmentation=None)
		self.config.IMAGE_RESIZE_MODE=RMODE
		fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2)
		ax1.imshow(np.max(IM,axis=2))
		ax1.set_title("Input")
		ax1.axis('off')
		print("Input shape = ", IM.shape)
		print("Input dtype = ", IM.dtype)
		ax2.imshow(M[:,:,ch])
		ax2.set_title("GT Mask")
		ax2.axis('off')
		print("GT Mask shape = ", M.shape)
		print("GT Mask dtype = ", M.dtype)
		ax3.imshow(X[:,:,ch])
		ax3.set_title("GT X-grad")
		ax3.axis('off')
		print("GT X-grad shape = ", X.shape)
		print("GT X-gray dtype = ", X.dtype)
		ax4.imshow(Y[:,:,ch])
		ax4.set_title("GT Y-grad")
		ax4.axis('off')
		print("GT Y-grad shape = ", Y.shape)
		print("GT Y-gray dtype = ", Y.dtype)
		plt.show()
		import pdb;pdb.set_trace()


	def train(self):

		"""Train the model."""

		augmentation = True
		print("TRAINING ALL LAYERS for {} epochs".format(self.config.EPOCHS))

		print("\n Warning: A large MSE will result at first, as the network begins to find the range for the gradients.")

		rlrop = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10)
		stop_on_nan = TerminateOnNaN()
		print_rl = MyCallback()
		reg_loss = RegLossCallback()
		#import pdb;pdb.set_trace()
		self.model.train(self.dataset_train, self.dataset_val,
					learning_rate=self.config.LEARNING_RATE,
					epochs=int(self.config.EPOCHS),
					augmentation=augmentation,
					layers='all',
					custom_callbacks=[rlrop, print_rl, stop_on_nan, reg_loss])

	def MouseTrain(self):
		"""
		Different training strategy.
		"""

		"""Train the model."""

		augmentation = True

		rlrop = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10)
		stop_on_nan = TerminateOnNaN()
		print_rl = MyCallback()

		print("TRAINING ALL LAYERS for {} epochs".format(self.config.EPOCHS))

		print("\n Warning: A large MSE will result at first, as the network begins to find the range for the gradients.")

		print("\nBeginning training, with lr at 0.2 and 50 iterations for numerical stability for one epoch...\n")

		#import pdb;pdb.set_trace()
		self.model.train(self.dataset_train, self.dataset_val,
					learning_rate=self.config.LEARNING_RATE,
					epochs=1,
					augmentation=augmentation,
					layers='all',
					train_it=50,
					custom_callbacks=[rlrop, print_rl, stop_on_nan])

		print("\nBeginning training, annealing learning rate from 0 to 0.2 over ten epochs...")
		for lr in np.linspace(0,self.config.LEARNING_RATE,10):
			#reset learning rate
			self.model.keras_model.optimizer.learning_rate.assign(lr)
			print(self.model.keras_model.optimizer.learning_rate)
			#tfKL.set_value(model.keras_model.optimizer.learning_rate, 0.2)

			self.model.train(self.dataset_train, self.dataset_val,
						learning_rate=lr,
						epochs=1,
						augmentation=augmentation,
						layers='all',
						custom_callbacks=[print_rl, stop_on_nan])

		print("\nTraining, with constant learning rate of 0.2 for 390 epochs...")
		self.model.keras_model.optimizer.learning_rate.assign(0.2)
		self.model.train(self.dataset_train, self.dataset_val,
					learning_rate=self.config.LEARNING_RATE,
					epochs=390,
					augmentation=augmentation,
					layers='all',
					custom_callbacks=[rlrop, print_rl, stop_on_nan])

		print("\nFinal training, with learning rate reduced by factor of two every 10 epochs for 100 epochs...")
		for lr in np.array([self.config.LEARNING_RATE]*10)/np.array([2,4,8,16,32,64,128,256,512,1024]):
			self.model.keras_model.optimizer.learning_rate.assign(lr)
			self.model.train(self.dataset_train, self.dataset_val,
						learning_rate=lr,
						epochs=10,
						augmentation=augmentation,
						layers='all',
						custom_callbacks=[print_rl, stop_on_nan])

	def run_train(self):
		if self.mode!='training':
			print("As a safety check, please reset the mode of the class object to 'training'.")
			return
		#create model instance if it does not yet exist.
		if not hasattr(self, 'model'):
			print("You need to create the basic model first. Please run 'create_model'")
			return

		#import data if it does not exist.
		if not hasattr(self, 'dataset_train'):
			print("You need to import the training data first. Please run 'import_train_val_data'")
			return

		if self.data_type=="MouseVision" or self.data_type=="MouseVisionSplit":
			self.MouseTrain()
		else:
			self.train()


	########################################################################
	########################################################################
	"""
	INFERENCE RELATED FUNCTIONS
	"""
	########################################################################
	########################################################################
	def import_inference_dataset(self):
		print("Prepping dataset...")
		dataset_test = copy.deepcopy(self.dataset)
		dataset_test.load_cell(self.dataset_dir)
		dataset_test.prepare()
		self.dataset_test = dataset_test
		print("Complete.")
		if self.config.AVG_PIX is None:
			if self.data_type=="Cell3D":
				print("With 3D models, it is recommended that you set the average pixel value for this set of images using config.AVG_PIX")
		if self.config.AVG_PIX is not None:
			if selc.data_type=="Cell2D":
				print("With 2D models, it is recommended that you do not set the average pixel value for this set of images using config.AVG_PIX = None")

	def see_inference_example(self, image_id=0, min_size=None):
		print("This function is deprecated.")
		#This doesn't quite work if the image is too big, since a padding is done.
		#really, this function isn't needed, so I'm deprecating it.
		IM, _ = utils.load_image_inference(self.dataset_test, self.config, image_id, max_size=1024, min_size=min_size)
		fig,ax1=plt.subplots(1)
		ax1.imshow(np.max(IM,axis=2))
		ax1.set_title("Inference image")
		ax1.axis('off')
		plt.show()
		import pdb;pdb.set_trace()

	def detect(self, save_dir, max_size=512, min_size=None, p_overlap=0.75, z_begin=None):
		assert max_size%2**(self.config.UNET_DEPTH-1)==0, "max_size argument in CellPose.detect must be divisible by 2 at least %d times" % self.config.UNET_DEPTH-1

		for image_id,_ in enumerate(self.dataset_test.image_info):
			#load image.
			print("Running inference on image: {}".format(os.path.join(self.dataset_test.image_info[image_id]['path'],self.dataset_test.image_info[image_id]['id'])))
			image,window = utils.load_image_inference(self.dataset_test, self.config, image_id, max_size, min_size=min_size, z_begin=z_begin)
			hmin,hmax,wmin,wmax=window

			need_tiling = [True if x>max_size else False for x in [hmax,wmax]]
			if any(need_tiling):
				#execute tiling.
				print("running image tiling and detection...")
				[tiles,im_tiles]=self.tile_image(image, tile_size=max_size, p_overlap=p_overlap)
				for i_tile,image in enumerate(im_tiles):
					print("Analyzing tile %d of %d" % (i_tile+1,len(im_tiles)))
					image = np.expand_dims(image, axis=0)
					[mask, xgrad, ygrad, edges, style]=self.model.keras_model.predict([image],verbose=0)
					xgrad /= 10.
					ygrad /= 10.
					tiles['image'].append(np.max(image[0,:,:],axis=2))
					tiles['mask'].append(mask[0,:,:])
					tiles['xgrad'].append(xgrad[0,:,:])
					tiles['ygrad'].append(ygrad[0,:,:])
					tiles['edges'].append(edges[0,:,:])
				print("Stitching detections...")
				[image,mask,xgrad,ygrad,edges]=self.stitch_image(max_size,tiles)
				#import pdb;pdb.set_trace()

				np.savez_compressed(os.path.join(save_dir,self.dataset_test.image_info[image_id]['id']),image=image, mask=mask, xgrad=xgrad, ygrad=ygrad, edges=edges)

			else:
				#need to add dimension for batch
				image = np.expand_dims(image, axis=0)
				#likely also need to pad probably.
				#run detection
				#
				"""
				This may work the first time through prediction, but after it is possible
				that the network will expect an image of the same shape!
				"""
				[mask, xgrad, ygrad, edges, style]=self.model.keras_model.predict([image],verbose=0)
				#import pdb;pdb.set_trace()
				image = image[0,hmin:hmax,wmin:wmax,:]
				#lets also save a max projection of the image.
				image = np.max(image,axis=2)
				mask = mask[0,hmin:hmax,wmin:wmax]
				xgrad = xgrad[0,hmin:hmax,wmin:wmax]
				ygrad = ygrad[0,hmin:hmax,wmin:wmax]
				edges = edges[0,hmin:hmax,wmin:wmax]
				#import pdb;pdb.set_trace()
				np.savez_compressed(os.path.join(save_dir,self.dataset_test.image_info[image_id]['id']),image=image, mask=mask, xgrad=xgrad, ygrad=ygrad, edges=edges)

	def tile_image(self, image, tile_size, p_overlap=0.75):
		"""
		Need a function to run detection on images which are too big.
		Goal: (1) Tile, hold positions
			  (2) Run detection on images.
			  These should be loaded into detect.
			  We either need to rework detect, or run_inference.
		Need to return "tiles" with keys (see stitch_image). but put in entry for locs.
		Make overlap at least 50% of the tile_size

		image should be size
		"""
		assert p_overlap<=1, "overlap percentage (decimal) must be less than 1"
		h_im, w_im = image.shape[:2]
		tile_size = int(tile_size)
		p_tile = int(np.ceil(tile_size*p_overlap))
		n_slices_vert = int(np.ceil(h_im / p_tile))
		n_slices_horz = int(np.ceil(w_im / p_tile))

		#Decide padding, if necessary.
		if h_im<n_slices_horz*p_tile:
			n_pad_rows = n_slices_vert*p_tile - h_im
		else:
			n_pad_rows = 0
		if w_im<n_slices_horz*p_tile:
			n_pad_cols = n_slices_horz*p_tile - w_im
		else:
			n_pad_cols = 0

		#apply padding to bottom row and last column only.
		Im_pad = copy.deepcopy(image)

		#THE PADDING HERE IS A PROBLEM IN DETECTION. It introduces a lot of noise in the detection.
		#We may instead want to end the last image differently. Or, mirror it.
		#break image into tiles.
		#reflect doesn't work, since in CellPose, you need to calculate gradients.
		#instead, we need to take last images differently.

		#determine start points for tiles to run detection on.
		#if doing padding, the next commented line works.
		#locs = [((ri,ri+tile_size),(ci,ci+tile_size)) for ri in range(0,h_im,p_tile) for ci in range(0, w_im, p_tile)]
		#the following code is for non-padded.
		locs=[]
		for ri in range(0,h_im,p_tile):
			for ci in range(0,w_im,p_tile):
				if ri+tile_size>h_im and ci+tile_size>w_im: #we are at the sets pf rows and columns
					locs.append(((h_im-tile_size,h_im),(w_im-tile_size,w_im)))
				elif ri+tile_size>h_im and ci+tile_size<=w_im: #we are at the last set of rows
					locs.append(((h_im-tile_size,h_im),(ci,ci+tile_size)))
				elif ri+tile_size<=h_im and ci+tile_size>w_im: #we are at the last set of columns
					locs.append(((ri,ri+tile_size),(w_im-tile_size,w_im)))
				else: #
					locs.append(((ri,ri+tile_size),(ci,ci+tile_size)))
		#There is overlap in these tiles. In CellPose, there is also edge effects when calculating
		#the gradients. Therefore, we want to only take the inner portions of each image.
		#Determine the acceptable windows of each tile image to use.
		trim_amount = int((1-p_overlap)*tile_size/2)
		#hmins = [ri + trim_amount if ri>0 else 0 for ri in range(0,h_im,p_tile)]#for ((ri,_),(_,_)) in locs]
		hmins = [0 if ri==0 else ri + trim_amount if ri + trim_amount <= h_im else ind*tile_size for ind,ri in enumerate(range(0,h_im,p_tile))]#for ((ri,_),(_,_)) in locs]
		#hmaxs = [ri + tile_size - trim_amount if ri + tile_size < h_im else ri + tile_size for ri in range(0,h_im,p_tile)]
		hmaxs = [ri + tile_size - trim_amount if ri + tile_size < h_im else h_im for ri in range(0,h_im,p_tile)]
		#wmins = [wi + trim_amount if wi>0 else 0 for wi in range(0, w_im, p_tile)]#for ((_,_),(wi,_)) in locs]
		wmins = [0 if wi==0 else wi + trim_amount if wi + trim_amount <= w_im else ind*tile_size for ind,wi in enumerate(range(0,w_im,p_tile))]#for ((ri,_),(_,_)) in locs]
		#wmaxs = [wi + tile_size - trim_amount if wi + tile_size < w_im else wi + tile_size for wi in range(0, w_im, p_tile)]
		wmaxs = [wi + tile_size - trim_amount if wi + tile_size < w_im else w_im for wi in range(0, w_im, p_tile)]

		#locs looks like list of each tile's (initial row, final row), (initial column, final column)
		#im_tiles = [Im_pad[ri:rend,ci:cend] for (ri,rend,ci,cend) in tiles['windows']]
		im_tiles = [Im_pad[x[0][0]:x[0][1],x[1][0]:x[1][1]] for x in locs]
		#import pdb;pdb.set_trace()

		rows = list(zip(hmins,hmaxs))
		cols = list(zip(wmins,wmaxs))
		import itertools
		windows = list(itertools.product(rows,cols))
		windows = [tuple(x[0] + x[1]) for x in windows]
		#create tiles dictionary
		tiles = {}
		#tiles['locs']=locs
		tiles['padding']=[n_pad_rows,n_pad_cols]
		tiles['init_shape']=(h_im,w_im)
		tiles['channels'] = self.config.OUT_CHANNELS
		tiles['dtype']=image.dtype
		tiles['trim']=trim_amount
		tiles['windows']=windows#list(zip(hmins,hmaxs,wmins,wmaxs)) #this is incorrect.
		tiles['image']=[]
		tiles['mask']=[]
		tiles['xgrad']=[]
		tiles['ygrad']=[]
		tiles['edges']=[]

		#import pdb;pdb.set_trace()
		#verify locs and windows line up. Checked, looks good.
		return tiles, im_tiles

	def stitch_image(self, tile_size, tiles):
		"""
		tiles should be a dictionary, with args 'windows','dtype', 'image', 'mask', 'xgrad', 'ygrad', 'edges', 'padding', 'init_shape'
		each key should be a list with tiles.
		How do we stitch together the predicted grads?
		Well, whereever the mask is on the border, we want to erase it.
		Therefore, use np.where and make the grads also zero wherever the mask is at on the border
		Then, just use max arg for edges.
		"""

		#for image, stitch together using max projection
		shape_to=tiles['init_shape']#[int(np.sum(el)) for el in zip(list(tiles['init_shape']),tiles['padding'])]
		image = np.zeros(shape=tuple(shape_to), dtype = tiles['dtype'])
		edges = np.zeros(shape=tuple(shape_to)+(tiles['channels'],), dtype = tiles['edges'][0].dtype)


		for tile_i in range(len(tiles['image'])):
			#x = tiles['locs'][tile_i]
			(ri,rend,ci,cend) = tiles['windows'][tile_i] #these are placements into image.

			if ri==0:
				rstart = 0
			elif rend==tiles['init_shape'][0]:
				rstart = tile_size-(rend-ri)
			else:
				rstart = tiles['trim']
			if ci==0:
				cstart = 0
			elif cend==tiles['init_shape'][1]:
				cstart=tile_size-(cend-ci)
			else:
				cstart = tiles['trim']
			if rend < tiles['init_shape'][0]:
				rstop = tile_size - tiles['trim']
			else:
				rstop = tile_size
			if cend < tiles['init_shape'][1]:
				cstop = tile_size - tiles['trim']
			else:
				cstop = tile_size
			#so to be clear, rstart,rstop,cstart,cstop are the indices for the tile.
			#import pdb;pdb.set_trace()
			if len(tiles['image'][tile_i].shape)==3:
				this_im = tiles['image'][tile_i][rstart:rstop,cstart:cstop,0] #note the 0 is just since there was an added channel.
			else:
				this_im = tiles['image'][tile_i][rstart:rstop,cstart:cstop]
			image[ri:rend,ci:cend] = this_im

			this_edge = tiles['edges'][tile_i][rstart:rstop,cstart:cstop,:]
			edges[ri:rend,ci:cend,:] = this_edge
			# im_temp = image[x[0][0]:x[0][1],x[1][0]:x[1][1]]
			# tile_im = tiles['image'][tile_i][:,:,0]
			# image[x[0][0]:x[0][1],x[1][0]:x[1][1]] = np.where(tile_im>im_temp,tile_im,im_temp)
			#
			# im_temp = edges[x[0][0]:x[0][1],x[1][0]:x[1][1]]
			# tile_im = tiles['edges'][tile_i]
			# edges[x[0][0]:x[0][1],x[1][0]:x[1][1]] = np.where(tile_im>im_temp,tile_im,im_temp)


		#for mask, remove objects that are on the border. For deleted pixels, delete the same Pixels
		#from xgrad and ygrad.
		mask = np.zeros(shape=tuple(shape_to)+(tiles['channels'],), dtype = tiles['mask'][0].dtype)
		xgrad = np.zeros(shape=tuple(shape_to)+(tiles['channels'],), dtype = tiles['xgrad'][0].dtype)
		ygrad = np.zeros(shape=tuple(shape_to)+(tiles['channels'],), dtype = tiles['ygrad'][0].dtype)

		#entered_array = np.zeros(shape=tuple(shape_to)) #keeps track of where pixels have been placed.

		#from skimage.measure import label
		#The way "clear_border" works is that it assumes 8-bit connectivity.
		#You can change that by first labeling the image
		#from skimage.segmentation import clear_border
		for tile_i in range(len(tiles['image'])):
			#x = tiles['locs'][tile_i]

			(ri,rend,ci,cend) = tiles['windows'][tile_i] #these are placements into image.

			if ri==0:
				rstart = 0
			elif rend==tiles['init_shape'][0]:
				rstart = tile_size-(rend-ri)
			else:
				rstart = tiles['trim']
			if ci==0:
				cstart = 0
			elif cend==tiles['init_shape'][1]:
				cstart=tile_size-(cend-ci)
			else:
				cstart = tiles['trim']
			if rend < tiles['init_shape'][0]:
				rstop = tile_size - tiles['trim']
			else:
				rstop = tile_size
			if cend < tiles['init_shape'][1]:
				cstop = tile_size - tiles['trim']
			else:
				cstop = tile_size

			#this_entered = entered_array[x[0][0]:x[0][1],x[1][0]:x[1][1]]

			# this_mask = tiles['mask'][tile_i]
			# this_x = tiles['xgrad'][tile_i]
			# this_y = tiles['ygrad'][tile_i]

			this_mask = tiles['mask'][tile_i][rstart:rstop,cstart:cstop,:]
			this_x = tiles['xgrad'][tile_i][rstart:rstop,cstart:cstop,:]
			this_y = tiles['ygrad'][tile_i][rstart:rstop,cstart:cstop,:]


			#The problem here is that the mask is not binary, you could threshold it, but why?
			#label_im = label(this_mask,connectivity=1)
			#clear_im = clear_border(label_im)
			##so now, we need to find the pixels that were deleted.
			#this_x=np.where(label_im>clear_im,0.0,this_x)
			#this_y=np.where(label_im>clear_im,0.0,this_y)



			#this_mask = np.where(clear_im>this_mask,clear_im,this_mask)

			mask[ri:rend,ci:cend,:] = this_mask
			xgrad[ri:rend,ci:cend,:] = this_x
			ygrad[ri:rend,ci:cend,:] = this_y

			#mask[x[0][0]:x[0][1],x[1][0]:x[1][1]]=np.where(this_mask > mask[x[0][0]:x[0][1],x[1][0]:x[1][1]], this_mask, mask[x[0][0]:x[0][1],x[1][0]:x[1][1]])
			"""
			lets try an average of the two, but we only want to average where there is data.
			Doing an average still has edge effects.
			Note, doing a max here doesn't work, since there are edge effects in the gradient estimation.
			Okay, the better estimation would be to reduce the accepted image data by 1-p_overlap / 2.
			So, in tile_image, we should create windows for each [hmin,hmax,wmin,wmax] tile image
			where hmin = max([0,locs[i][0][0]])
			This is in tiles['windows']

			NEED TO FINISH THIS!
			"""
			#xgrad[x[0][0]:x[0][1],x[1][0]:x[1][1]]=np.where(this_entered<1, this_x, (xgrad[x[0][0]:x[0][1],x[1][0]:x[1][1]] + this_x) / 2)
			#ygrad[x[0][0]:x[0][1],x[1][0]:x[1][1]]=np.where(this_entered<1, this_y, (ygrad[x[0][0]:x[0][1],x[1][0]:x[1][1]] + this_y) / 2)
			#entered_array[x[0][0]:x[0][1],x[1][0]:x[1][1]]=1


		image = image[0:tiles['init_shape'][0],0:tiles['init_shape'][1]]
		mask = mask[0:tiles['init_shape'][0],0:tiles['init_shape'][1],:]
		xgrad = xgrad[0:tiles['init_shape'][0],0:tiles['init_shape'][1],:]
		ygrad = ygrad[0:tiles['init_shape'][0],0:tiles['init_shape'][1],:]
		edges = edges[0:tiles['init_shape'][0],0:tiles['init_shape'][1],:]

		return image, mask, xgrad, ygrad, edges


	def run_inference(self, max_size=1024, min_size=None, p_overlap=0.75, submit_dir = None, z_begin = None):
		"""
		Use this when running on an entire directory.
		Arguments:
		max_size: Size of image to run inference on. If max_size is smaller than
				  source image in any dimension, a tiling algorithm will be used.
				  Reducing max_size uses a smaller computational load, but Results
				  in more images to be analyzed and thus a longer analysis.

		p_overlap: 1-p_overlap represents the percentage of overlap between consecutive tiles.

		submit_dir:
		"""
		print("Running inference. Note: If a segfault results, try reducing the inference size with argument 'max_size'.")
		if self.mode!='inference':
			print("As a safety check, please reset the mode of the class object to 'inference'.")
			return
		#create model instance if it does not yet exist.
		if not hasattr(self, 'model'):
			print("You need to first create a basic model. Run 'create_model'")
			return

		if not hasattr(self, 'dataset_test'):
			print("You need to first import the inference dataset. Run 'import_inference_dataset'")
			return

		# Create directory
		if not submit_dir:
			ROOT_DIR = os.path.abspath("./")
			RESULTS_DIR = os.path.join(ROOT_DIR, "results")
			if not os.path.exists(RESULTS_DIR):
				os.makedirs(RESULTS_DIR)
			submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
			submit_dir = os.path.join(RESULTS_DIR, submit_dir)
			os.makedirs(submit_dir)
		else:
			if not os.path.exists(submit_dir):
				os.makedirs(submit_dir)
		#import pdb;pdb.set_trace()
		self.detect(submit_dir, max_size, min_size, p_overlap, z_begin = z_begin)

	########################################################################
	########################################################################
	"""
	Metric calculation functions!
	"""
	########################################################################
	########################################################################

	def measure_metrics(self, mask_thresh=0.5, boundary_thresh=0.8, size_thresh=50, alpha=0.5, detection_dir = None):
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
		if self.mode!='inference':
			print("As a safety check, please reset the mode of the class object to 'inference'.")
			raise ValueError("Must be in 'inference' mode to run measure_metrics")
		if not hasattr(self, 'dataset_test'):
			print("You need to first import the inference dataset. Run 'import_inference_dataset'")
			raise ValueError("Must first import inference dataset to run measure_metrics")
		#check that the necessary files exist in filepath
		if not os.path.isdir(os.path.join(self.dataset_dir, "gt")):
			print("mask directory does not exist at: {}".format(os.path.join(self.dataset_dir, "gt")))
			raise ValueError("Mask directory does not exit.")
		###CHECK DETECTION DIRECTORY.
		if detection_dir is not None:
			if not os.path.isdir(os.path.join(self.dataset_dir, detection_dir)):
				if not os.path.isdir(detection_dir):
					raise ValueError("Detection directory does not exit.")
			else:
				detection_dir = os.path.join(self.dataset_dir, detection_dir)
		else:
			if not os.path.isdir(os.path.join(self.dataset_dir, "detections")):
				print("detection directory does not exist at: {}".format(os.path.join(self.dataset_dir, "detections")))
				raise ValueError("Detection directory does not exit.")
			else:
				detection_dir = os.path.join(self.dataset_dir, "detections")

		self.dataset_test.detect_dir = detection_dir
		#import pdb;pdb.set_trace()

		#if a single alpha value is given, turn to list.
		if isinstance(alpha,float):
			alpha = [alpha]

		#Load predicted masks and boundaries, gt masks and boundaries. Put each in lists.
		#So you should have loaded as an "inference" dataset. Therefore, all files are as:
		#CP.dataset_test.image_info[image_id] with the structure
		#{'id': '25A_15D_022020_Experiment002_Series004_t05',
		#'source': 'cell',
		#'path': '/Users/czeddy/Documents/Auto_Seg/shared_repo/CellPose/Datasets/v7_mini/train'}
		#so need to change path to end at v7 mini.
		####COPY the dataset so we can change the paths.
		#dataset_copy = copy.copy(self.dataset_test)
		# gt_masks = []
		# gt_borders = []
		# pred_masks = []
		# pred_borders = []
		all_ious = []
		all_intersections = []
		all_unions = []
		all_gt_mask_pix = []
		all_pred_mask_pix = []
		changed_boundary = False
		print("Importing images and calculating IoUs...")
		for i_info in range(len(self.dataset_test.image_info)):#dataset_copy.image_info)):
			#load images!
			print("Importing {}...".format(self.dataset_test.image_info[i_info]['id']))
			#gt_mask, _, _, gt_border, pred_mask, _, _, pred_border = utils.load_image_gt_metrics(self.dataset_test, self.config, i_info)
			gt_mask, gt_border, pred_mask, pred_border, return_flag = utils.load_image_gt_metrics(self.dataset_test, self.config, i_info)
			if return_flag:
				changed_boundary = True
				boundary_thresh = 10000. #in this case, the results npz file was loaded, so we don't want to remove boundaries.
			#there is a possibility that you could have loaded a 2D model when detection
			#was done with a 3D model, or vice versa. Check that it passes the test.
			# if gt_mask.shape[2] != pred_mask.shape[2]:
			# 	if gt_mask.shape[2] > pred_mask.shape[2]:
			# 		print("\n****The model dimensionality is 3D when predictions were ran with a 2D network.****\n")
			# 		raise ValueError("You must set the model dimensionality to 2D for this dataset.")
			# 	else:
			# 		print("\n****The model dimensionality is 2D when predictions were ran with a 3D network.****\n")
			# 		raise ValueError("You must set the model dimensionality to 3D for this dataset.")

			# #store in lists
			# gt_masks.append(gt_mask)
			# gt_borders.append(gt_border)
			# pred_masks.append(pred_mask)
			# pred_borders.append(pred_border)
			# #the gt are a lot to store, since they are H x W x N # detections
			# #6 images takes 0.5 gB memory, we'll have 100 images, which adds to 8.33 gB memory required.
			# #other alternatives would be to store the pixels of the object and
			# #rebuild the image each time.
			# #Or do the thresholding at this point and the mask border subtraction.
			# #then redo the iou so the intersection is any rows they have in common,
			# #the union is the sum of unique pixel combos. Consider this if you do not
			# #have the available memory.
			#calculate the region pixels.
			gt_mask_pix, pred_mask_pix = metrics.get_image_regions(pred_mask, pred_border, gt_mask, gt_border, mask_thresh=mask_thresh, boundary_thresh=boundary_thresh, size_thresh=size_thresh)
			#calculate the ious of the different regions, return iou array.
			iou_matrix, intersections, unions = metrics.calculate_IoUs_v2(gt_mask_pix, pred_mask_pix)
			#import pdb;pdb.set_trace()
			#store the iou matrix
			all_ious.append(iou_matrix)
			all_intersections.append(intersections)
			all_unions.append(unions)
			all_gt_mask_pix.append(gt_mask_pix)
			all_pred_mask_pix.append(pred_mask_pix)
			#import pdb;pdb.set_trace()

		if not changed_boundary:
			print("Using raw predictions of cell/background and edges to form cell labels.")
			print("Using processed files for comparison to ground truths will lead to better precision estimations.")
			print("It is recommended that you first use process_results on the detections directory.")
		#import pdb;pdb.set_trace()
		# #calculate the intersections and unions.
		# print("Calculating IoU values of predictions vs ground truth...")
		# all_intersections, all_unions, all_ious = metrics.calculate_IoUs(pred_masks,
		# 	pred_borders, gt_masks, gt_borders, mask_thresh=mask_thresh,
		# 	boundary_thresh=boundary_thresh, size_thresh=size_thresh)

		all_matches = []
		all_image_AP = []
		zero_matches = metrics.pred_to_gt_assignment(all_ious, alpha=0.)
		for i,min_iou in enumerate(alpha):
			print("\n Analyzing metrics with min IoU of {}".format(min_iou))
			matches = metrics.pred_to_gt_assignment(all_ious, alpha=min_iou)
			#images_AP = metrics.calculate_average_precision(matches, all_ious, all_intersections, all_unions, all_gt_mask_pix, all_pred_mask_pix)
			M = [np.where(y>=min_iou, x, False) for (x,y) in zip(zero_matches,all_ious)] #this way, the matching doesn't change. I wonder how it compares to matches
			images_AP = metrics.calculate_average_precision(M, all_ious, all_intersections, all_unions, all_gt_mask_pix, all_pred_mask_pix)
			#this average precision is calculated as a
			#a 1D array of length N images.
			all_image_AP.append(images_AP)
			all_matches.append(matches)

		qualities = metrics.get_matched_mask_qualities(all_matches, all_ious)

		#get CellPose metrics
		tp, fp, fn, ap = metrics.get_cellpose_precision(all_matches)
		#ap = tp / (tp + fp + fn)

		#PUT INTO A FUNCTION IN METRICS.PY
		#calculate P vs R curve.
		#CANNOT DO THIS! See metrics.py function calculate_metrics for details.
		# print("Evaluating the mean average precision over {} images.".format(len(all_ious)))
		#
		# #CALCULATE mAP
		# mAPs = np.zeros(len(alpha))
		# for i in range(len(alpha)):
		# 	mAPs[i] = metrics.calculate_mean_average_precision(np.append([0],all_precision_data[i]),np.append([0],all_recall_data[i]))
		# #import pdb;pdb.set_trace()
		# fig,ax = plt.subplots(1)
		# for i in range(len(all_precision_data)):
		# 	ax.plot(np.append(all_recall_data[i],[all_recall_data[i][-1]]),np.append(all_precision_data[i],[0.]),'-',label='mAP_%d = %.3f'%(int(alpha[i]*100),mAPs[i]))
		# ax.set_xlabel("Recall")
		# ax.set_ylabel("Precision")
		# ax.legend()
		# plt.show()

		#wrap items into a nice dictionary
		results = {}
		for variable in ["all_ious", "all_matches", "zero_matches", "all_image_AP", "qualities", "tp", "fp", "fn", "ap"]:
			results[variable] = eval(variable)

		return results


	########################################################################
	########################################################################
	"""
	VISUALIZE RELATED FUNCTIONS
	"""
	########################################################################
	########################################################################


	####FUNCTIONS FOR STEM OUTPUT COMPARISONS#####

	def create_stem_model(self):
		#Load different network
		self.vis_model = modellib.StemNet(self.config)

		#if given, load weights.
		if self.weights:
			modellib.load_weights(self.vis_model.keras_model, self.weights)

		print("Printing Stem Model layers...")
		self.vis_model.keras_model.summary()

	def visualize_stem_out(self, model, dataset, image_id=0, max_size=1024):
		"""
		Visualize the output from stem (or just input) into network
		Using dataset, will analyze image idx.
		"""
		print("Visualizing stem output on image: {}".format(os.path.join(dataset.image_info[image_id]['path'],dataset.image_info[image_id]['id'])))
		image,window = utils.load_image_inference(dataset, self.config, image_id, max_size=max_size)
		hmin,hmax,wmin,wmax=window
		#need to add dimension for batch
		image = np.expand_dims(image, axis=0)
		if model is not None:
			if model.config.INPUT_DIM=="3D":
				[image_out]=model.keras_model.predict([image],verbose=0)
			else:
				image_out = image
		else:
			image_out=image
		return image, image_out

	def run_stem_visualize(self, image_id=0, max_size=1024):
		import tifffile

		if self.mode!='visualize':
			print("As a safety check, please reset the mode of the class object to 'visualize'.")
			return

		if not hasattr(self,'dataset_test'):
			print("You must first load dataset. Run 'import_inference_dataset'")
			return

		if hasattr(self.config, 'INPUT_DIM'):
			if self.config.INPUT_DIM=="3D":
				if not hasattr(self,'vis_model'):
					print("You must first create Stem model. Run 'create_stem_model'")
					return
				IM, IM_stem = self.visualize_stem_out(self.vis_model, self.dataset_test, image_id, max_size=max_size)
			else:
				print("\nModel is a 2D input model, therefore there is no stem to visualize.")
				print("Instead, returning input image.\n")
				IM, IM_stem = self.visualize_stem_out(None, self.dataset_test, image_id, max_size=max_size)
		else:
			print("\nModel is a 2D input model, therefore there is no stem to visualize.")
			print("Instead, returning input image.\n")
			IM, IM_stem = self.visualize_stem_out(None, self.dataset_test, image_id, max_size=max_size)

		tifffile.imwrite(os.path.join(os.path.dirname(self.weights),self.dataset_test.image_reference(image_id)+'_StemOut.tif'),IM_stem,photometric='rgb')
		tifffile.imwrite(os.path.join(os.path.dirname(self.weights),self.dataset_test.image_reference(image_id)+'_StemOut_R.tif'),IM_stem[:,:,0])
		tifffile.imwrite(os.path.join(os.path.dirname(self.weights),self.dataset_test.image_reference(image_id)+'_StemOut_G.tif'),IM_stem[:,:,1])
		tifffile.imwrite(os.path.join(os.path.dirname(self.weights),self.dataset_test.image_reference(image_id)+'_StemOut_B.tif'),IM_stem[:,:,2])
		#import matplotlib.pyplot as plt
		# fig, (ax1,ax2,ax3)=plt.subplots(1,3)
		# ax1.imshow(IM_stem[:,:,0])
		# ax2.imshow(IM_stem[:,:,1])
		# ax3.imshow(IM_stem[:,:,2])
		# plt.show()

	def visualize_stem_padcheck(self, model, dataset, image_id=0, max_size=1024):
		"""
		Visualize the output from stem (or just input) into network
		Using dataset, will analyze image idx.
		Tries to do all padding options at once.
		"""
		assert model.config.INPUT_DIM=="3D", "Model must be set for 3D."
		IM = dataset.load_image(image_id, dimensionality=model.config.INPUT_DIM, avg_pixel=0.0)
		data_width = IM.shape[2]
		#clear IM from memory.
		del IM

		padding_options = [z for z in range(model.config.INPUT_Z - data_width)]
		padded_images = np.stack([utils.load_image_PadCheck(dataset, self.config, image_id, max_size=max_size, z_before = z) for z in range(model.config.INPUT_Z - data_width)],axis=0)
		#above adds dimension as batch
		#run prediction.
		if model is not None:
			if model.config.INPUT_DIM=="3D":
				print("Running PEN prediction on {} images...".format(len(padding_options)))
				images_out=model.keras_model.predict([padded_images],verbose=0)
			else:
				images_out = image
		else:
			images_out=image
		#every image should be an HxWx3 image.
		return padding_options,images_out

	def visualize_stem_padcheck_slower(self, model, dataset, image_id=0, max_size=1024,together=1):
		"""
		Visualize the output from stem (or just input) into network
		Using dataset, will analyze image idx.
		Runs inference on a subset of images at a time, slower, but uses far less memory.
		together = how many images to run simultaneously, more memory but faster run time.
		"""
		assert model.config.INPUT_DIM=="3D", "Model must be set for 3D."
		IM = dataset.load_image(image_id, dimensionality=model.config.INPUT_DIM, avg_pixel=0.0)
		(H,W,data_width) = IM.shape[:3]
		#creating images for analysis...
		padding_options = [z for z in range(model.config.INPUT_Z - data_width)]
		#padded_images = np.stack([utils.load_image_PadCheck(dataset, self.config, image_id, max_size=max_size, z_before = z) for z in range(model.config.INPUT_Z - data_width)],axis=0)
		#above adds dimension as batch
		#run prediction.
		output_ims=np.zeros(shape=(len(padding_options),H,W,3),dtype=IM.dtype)
		#clear IM from memory.
		del IM
		if model is not None:
			if model.config.INPUT_DIM=="3D":
				for n in range(0,len(padding_options),together):
					print("Running PEN prediction on image(s) {} of {}...".format([x+1 for x in list(range(n,min(n+together,len(padding_options))))],len(padding_options)))
					these_pads = padding_options[n:min(n+together,len(padding_options))]
					padded_images = np.stack([utils.load_image_PadCheck(dataset, self.config, image_id, max_size=max_size, z_before = z) for z in these_pads],axis=0)
					#above added dimension for batch Already
					#for some models that didn't have input images properly normalized need to 0-1, do the following:
					#padded_images*=255.
					#import pdb;pdb.set_trace()
					output=model.keras_model.predict([padded_images],verbose=0)
					output_ims[n:min(n+together,len(padding_options)),:,:,:]=output

		#every image should be an HxWx3 image.
		return padding_options,output_ims

	def stem_padcheck(self, image_id=0, max_size=1024):
		print("Visualizing padding options on image: {}".format(os.path.join(self.dataset_test.image_info[image_id]['path'],self.dataset_test.image_info[image_id]['id'])))
		if self.mode!='visualize':
			print("As a safety check, please reset the mode of the class object to 'visualize'.")
			return

		if not hasattr(self,'dataset_test'):
			print("You must first load dataset. Run 'import_inference_dataset'")
			return

		if hasattr(self.config, 'INPUT_DIM'):
			if self.config.INPUT_DIM=="3D":
				if not hasattr(self,'vis_model'):
					print("You must first create Stem model. Run 'create_stem_model'")
					return
				opts, IMs_stem = self.visualize_stem_padcheck_slower(self.vis_model, self.dataset_test, image_id, max_size=max_size, together=3)
				return opts, IMs_stem


	def pull_layer_names(self):
		"""
		Note: you need to run "create_model()" before this
		"""
		if hasattr(self,'model'):
			model_layers = [a.name for a in self.model.keras_model.layers]
		return model_layers

	def find_ind_layer(self, entry, model_layers):
		indices = [i for i, elem in enumerate(model_layers) if entry in elem]
		return indices

	def pull_layer_weights(self, layer_i, model_layers):

		if layer_i>-1 and layer_i<len(model_layers):
			weights = self.model.keras_model.layers[layer_i].get_weights()
			if weights:
				print("returning weights and bias from layer %s" % model_layers[layer_i])
				return weights
			else:
				print("layer %s does not have any learned parameters" % model_layers[layer_i])
		else:
			print("layer_i argument is not an index within the model layers")

	def visualize_kernel_2d(self, weights, filter_n, input_channel):
		#not really a good visual process for 3D kernels.
		if len(weights[0].shape)==5:
			print("weights are from a 3D kernel. This function is not recommended.")
		F = np.zeros(shape=weights[0].shape[0:2])
		F[:,:] = weights[0][:,:,input_channel,filter_n]
		plt.imshow(F, cmap='gray')
		plt.show()



######################################################################
"""
Post-processing Scripts
In Inference, we find the gradients, masks, and store the projected image into an .npz file.
Here, we translate those to seeds (centroid object locations), and labeled masks.
Again, we store the data as a compressed .npz file.
Finally, other functions are used for visualizing results.
Best method is to combine the saved tiff file from visualize_labels with the max-
projection of your input image using ImageJ Overlay function with 55 opacity and zero transparency.
"""
#####################################################################

def process_result(filepath, centers_iter=500, masks_iter=300, mask_thresh=0.5, border_thresh=0.8, grad_scale=1., verbose=False):
	"""
	Run script from flows_to_masks.py
	Generates "seeds": roughly center flow points, and "labels": cell masks.
	Saves as an .npz file.
	Recommend only applying a gradient threshold IF you must speed things up
	"""
	flows_to_masks.npz_to_mask(filepath, centers_iter, masks_iter, mask_thresh, border_thresh, verbose=verbose)

def export_all_labels_to_mat(f_path):
	import glob
	fpaths = glob.glob(f_path+"/*_results.npz", recursive=False)
	for pth in fpaths:
		export_labels_to_mat(pth)

def export_labels_to_mat(f_path):
	"""
	Load npz saved from process_result, and save label data as a .mat Octave/Matlab file
	"""
	import scipy.io
	loaded = np.load(f_path)
	labels = loaded['label']
	scipy.io.savemat(f_path[:f_path.rfind(".")]+".mat", dict(labels=labels))

def compare_pred_vs_gt(json_path, f_path):
	"""
	Only for results in validation set.
	f_path: path to npz file from process_result
	json_path: path to ground truth JSON file.

	Compare automated detection to ground truth data. Requires JSON file annotations.
	"""
	import skimage
	from skimage.morphology import erosion, square
	from matplotlib.colors import Normalize
	# modify the default parameters of np.load
	temp_loader = lambda *a,**k: np.load(*a, allow_pickle=True, **k)
	loaded = temp_loader(f_path[:f_path.rfind(".")]+"_results.npz")
	loaded_prior = np.load(f_path)
	seeds = loaded['seeds']
	seeds = np.concatenate(seeds,axis=0)
	labels = loaded['label']
	IM=loaded_prior['image']

	data = utils.load_json_data(json_path)
	mask = np.zeros([data["images"]["height"], data["images"]["width"], len(data['annotations']['regions']['area'])],
					dtype=np.uint8)
					#puts each mask into a different channel.
	for i,[verty,vertx] in enumerate(zip(data['annotations']['regions']['x_vert'],data['annotations']['regions']['y_vert'])):
		#alright, so this appears backwards (vertx, verty) but it is this way because of how matplotlib does plotting.
		#I have verified this notation is correct CE 11/20/20
		poly = np.transpose(np.array((vertx,verty)))
		rr, cc = skimage.draw.polygon(poly[:,0], poly[:,1], mask.shape[0:-1])
		RR, CC = skimage.draw.polygon_perimeter(poly[:,0], poly[:,1], mask.shape[0:-1])
		#In load_image_inference, bad_pixels is used to pick up all potential object pixels. In load_image when the "mask" argument is passed,
		# if there exists an annotated object, then the pixels of bad_pixels are deleted. In load_image_inference, there is no mask argument
		#Just fixed this issue CE 12/15/21 in utils load_image
		if np.sum(IM[rr,cc])==0:
			continue
		try:
			mask[rr,cc,i] = 1
		except:
			print("too many objects, needs debugging")
			print(self.image_info[image_id])

	border_image = np.zeros([data["images"]["height"], data["images"]["width"]])
	val=0
	for n in range(mask.shape[2]):
		val+=1
		this_mask = mask[:,:,n]
		#plan: Erode, invert, multiply.
		inner_mask = erosion(this_mask,square(3))
		inner_mask = inner_mask<0.5 #should be binary anyway, so this is fine.
		border = inner_mask * this_mask
		border[border>0]=val
		border_image = np.where(border>0,border,border_image)

	#alpha_map = border_image>0
	#alpha_map = alpha_map.astype('float')
	#import pdb;pdb.set_trace()

	init_border_image = np.stack([border_image,border_image,border_image],axis=2)
	final_border_image = np.stack([border_image,border_image,border_image],axis=2)
	for n in range(1,int(np.max(np.unique(border_image))+1)):
		r,g,b=random_bright_color()
		final_border_image = np.where(init_border_image==n, [r,g,b], final_border_image)

	final_border_image = final_border_image.astype(int) #conver to int, since range is 0-255


	init_label_image = np.stack([np.max(labels,axis=2)]*3,axis=2)
	final_label_image = np.stack([np.max(labels,axis=2)]*3,axis=2)
	for slice in range(labels.shape[2]):
		for obj in range(1, int(np.max(np.unique(labels[:,:,slice]))+1)):
			r,g,b=random_bright_color()
			final_label_image = np.where(init_label_image==obj, [r,g,b], final_label_image)

	final_label_image = final_label_image.astype(int) #conver to int, since range is 0-255
	#rescale.
	if len(IM.shape)>2:
		IM=np.squeeze(np.max(IM,axis=2))
	IM = IM/255.0
	#if np.max(IM)<=1.0:
	#	IM *= 255

	final_label_image = np.where(final_border_image>0, final_border_image, final_label_image)
	#alphas = np.clip(final_label_image, 0.01, 1)  # alpha value clipped at the bottom at .4
	fig,(ax1,ax2) = plt.subplots(1,2)
	ax1.imshow(np.stack([IM,IM,IM],axis=2))
	ax1.imshow(final_label_image,alpha=0.5)
	#plt.imshow(np.stack([final_border_image,alpha_map],axis=2))
	ax1.plot(seeds[:,1],seeds[:,0],'rx')

	ax2.imshow(np.stack([IM,IM,IM],axis=2))
	plt.show()

	flows_to_masks.write_to_image(final_label_image,os.path.dirname(f_path),"gt_"+os.path.basename(f_path))

	#import pdb;pdb.set_trace()

def save_cell_borders(json_path):
	"""
	Only for results in validation set.
	json_path: path to ground truth JSON file.

	Produce RGB image of JSON file annotation edges.
	"""
	import skimage
	from skimage.morphology import erosion, square
	from matplotlib.colors import Normalize
	# modify the default parameters of np.load
	temp_loader = lambda *a,**k: np.load(*a, allow_pickle=True, **k)
	loaded = temp_loader(f_path)#[:f_path.rfind(".")]+"_results.npz")
	seeds = loaded['seeds']
	seeds = np.concatenate(seeds,axis=0)
	labels = loaded['label']

	data = utils.load_json_data(json_path)
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

	border_image = np.zeros([data["images"]["height"], data["images"]["width"]])
	val=0
	for n in range(mask.shape[2]):
		val+=1
		this_mask = mask[:,:,n]
		#plan: Erode, invert, multiply.
		inner_mask = erosion(this_mask,square(3))
		inner_mask = inner_mask<0.5 #should be binary anyway, so this is fine.
		border = inner_mask * this_mask
		border[border>0]=val
		border_image = np.where(border>0,border,border_image)

	init_border_image = np.stack([border_image,border_image,border_image],axis=2)
	final_border_image = np.stack([border_image,border_image,border_image],axis=2)
	for n in range(1,int(np.max(np.unique(border_image))+1)):
		r,g,b=random_bright_color()
		final_border_image = np.where(init_border_image==n, [r,g,b], final_border_image)

	final_border_image = final_border_image.astype(int) #conver to int, since range is 0-255

	flows_to_masks.write_to_image(final_border_image,os.path.dirname(f_path),"gt_borders_only_"+os.path.basename(f_path))

def visualize_seeds_and_labels(f_path_a, f_path_b=None, show_plot=True):
	"""
	Load npz saved from process_result
	See cell centers and labels.
	"""
	# modify the default parameters of np.load
	temp_loader = lambda *a,**k: np.load(*a, allow_pickle=True, **k)
	if f_path_b:
		loaded = temp_loader(f_path_b)
	else:
		loaded = temp_loader(f_path_a[:f_path_a.rfind(".")]+"_results.npz")
	loaded_prior = np.load(f_path_a)
	seeds = loaded['seeds']
	seeds = np.concatenate(seeds,axis=0)
	labels = loaded['label']
	(h,w) = labels.shape[:2]
	red = np.zeros(shape=(h,w),dtype=np.float32)
	green = np.zeros(shape=(h,w),dtype=np.float32)
	blue = np.zeros(shape=(h,w),dtype=np.float32)
	d = np.zeros(shape=(h,w),dtype=np.float32)
	for ch in range(labels.shape[2]):
		for _,obj in enumerate(np.unique(labels[:,:,ch])):
			if obj>0.:
				r,g,b = random_bright_color()
				#c = np.random.rand()
				d[:]=r
				red = np.where(labels[:,:,ch]==obj, d, red)
				#c = np.random.rand()
				d[:]=g
				green = np.where(labels[:,:,ch]==obj, d, green)
				#c = np.random.rand()
				d[:]=b
				blue = np.where(labels[:,:,ch]==obj, d, blue)

	recolor=np.stack([red,green,blue],axis=2)
	#find unique values in labels.
	#labels could be multiple channels.
	IM=loaded_prior['image']
	#rescale.
	if len(IM.shape)>2:
		IM=np.squeeze(np.max(IM,axis=2))
	IM = IM/255.0
	recolor = recolor / 255.0
	if show_plot:
		plt.imshow(np.stack([IM,IM,IM],axis=2))
		plt.imshow(recolor,alpha=0.5)
		plt.plot(seeds[:,1],seeds[:,0],'rx')
		plt.show()
		return
	else:
		return IM, recolor, seeds

def visualize_seeds(f_path_a, f_path_b=None):
	"""
	Load npz files
	file_path_a: path to file output from running CellPose.run_inference()
	file_path b: path to file output after running process_result
	NOTE: If the centers appear not to be ON the cell at all, it is because
	in flows_to_masks.py (functions get_masks and steps2D_fast, see 'Po += ' bits

	See cell centers only.
	"""
	temp_loader = lambda *a,**k: np.load(*a, allow_pickle=True, **k)
	loaded = np.load(f_path_a)
	IM = loaded['image']
	if len(IM.shape)>2:
		IM = np.max(IM,axis=2)
	IM = np.stack([IM,IM,IM],axis=2)

	if f_path_b:
		loaded = temp_loader(f_path_b)
	else:
		loaded = temp_loader(f_path_a[:f_path_a.rfind(".")]+"_results.npz")
	seeds = loaded['seeds']
	seeds = np.concatenate(seeds, axis=0)
	plt.imshow(IM)
	plt.plot(seeds[:,1],seeds[:,0],'rx')
	plt.show()

def visualize_labels(f_path_a, f_path_b=None):
	"""
	Load npz files
	file_path_a: path to file output from running CellPose.run_inference()
	file_path b: path to file output after running process_result
	NOTE: If the centers appear not to be ON the cell at all, it is because
	in flows_to_masks.py (functions get_masks and steps2D_fast, see 'Po += ' bits

	See cell labels.
	"""
	loaded = np.load(f_path_a)
	IM = loaded['image']
	if len(IM.shape)>2:
		IM = np.max(IM,axis=2)
	IM = np.stack([IM,IM,IM],axis=2)
	BW = np.zeros_like(IM)

	if f_path_b:
		loaded = np.load(f_path_b)
	else:
		loaded = np.load(f_path_a[:f_path_a.rfind(".")]+"_results.npz")
	labels = loaded['label']
	#so, we could arbitrarily decide a color now.

	all_labs = np.unique(labels)
	counter=0
	for ch in range(labels.shape[2]):
		#pull unique labels.
		lab_i = np.unique(labels[:,:,ch])
		for i,obj in enumerate(lab_i):
			counter+=1
			print("\r%.1f percent complete" % float(counter/(len(all_labs)+1)*100), end='',flush=True)
			if obj>0.0:
				this_mask = labels[:,:,ch]==obj
				r,g,b=random_bright_color()
				IM[:,:,0]=np.where(this_mask,r,IM[:,:,0])
				IM[:,:,1]=np.where(this_mask,g,IM[:,:,1])
				IM[:,:,2]=np.where(this_mask,b,IM[:,:,2])
				BW[:,:,0]=np.where(this_mask,r,BW[:,:,0])
				BW[:,:,1]=np.where(this_mask,g,BW[:,:,1])
				BW[:,:,2]=np.where(this_mask,b,BW[:,:,2])
	print("")

	flows_to_masks.write_to_image(IM,os.path.dirname(f_path_a),os.path.basename(f_path_a))
	flows_to_masks.write_to_image(BW,os.path.dirname(f_path_a),"detect_"+os.path.basename(f_path_a))
	#plt.imshow(IM)
	#plt.show()

def process_visualize_directory(f_path):
	import glob
	#glob all correct filenames
	if f_path[-1]!="/":
		f_path = f_path+"/"
	fname_list = glob.glob(f_path+"*.npz")
	fname_list = [x for x in fname_list if "results" not in os.path.basename(x)]
	fname_list = [x for x in fname_list if os.path.basename(x)[0]!="."]
	for fname in fname_list:
		process_result(fname, verbose=True)
		visualize_labels(fname)

def process_directory(f_path):
	import glob
	#glob all correct filenames
	if f_path[-1]!="/":
		f_path = f_path+"/"
	fname_list = glob.glob(f_path+"*.npz")
	fname_list = [x for x in fname_list if "results" not in os.path.basename(x)]
	fname_list = [x for x in fname_list if os.path.basename(x)[0]!="."]
	for i,fname in enumerate(fname_list):
		print("Image {}/{}".format(i+1,len(fname_list)))
		process_result(fname, verbose=True)

def process_directory_parallel(f_path, fraction=1.):
	"""
	INPUTS
	--------------------------------------
	f_path = directory to stored .npz files in which to run process_file on.

	fraction = float (<= 1.) which specifies the fraction of available cores to
			   run parallel processes on.
	"""
	import glob
	from joblib import Parallel, delayed
	import multiprocessing
	#glob all correct filenames
	if f_path[-1]!="/":
		f_path = f_path+"/"
	fname_list = glob.glob(f_path+"*.npz")
	fname_list = [x for x in fname_list if "results" not in os.path.basename(x)]
	fname_list = [x for x in fname_list if os.path.basename(x)[0]!="."]
	num_cores = multiprocessing.cpu_count()
	num_cores_to_use = max(1,int(np.round(num_cores*fraction)))
	Parallel(n_jobs = num_cores_to_use)(delayed(process_result)(fname) for fname in fname_list)


def visualize_edges(f_path):
	"""
	load npz from inference
	"""
	loaded = np.load(f_path)
	IM = loaded['image']
	if len(IM.shape)>2:
		IM = np.max(IM,axis=2)
	IM = np.stack([IM,IM,IM],axis=2)
	edges = loaded['edges']
	edges=np.stack([edges,np.zeros(shape=edges.shape,dtype=edges.dtype),np.zeros(shape=edges.shape,dtype=edges.dtype)],axis=2)

	IM_out = np.where(np.abs(edges)>0.0,np.abs(edges),IM)
	plt.imshow(IM_out)
	plt.show()

def visualize_gradients(fpath):
	"""
	load npz from inference
	"""
	loaded = np.load(fpath)
	xgrad = loaded['xgrad']
	ygrad = loaded['ygrad']
	edge = loaded['edges']
	mask = loaded['mask']
	fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2)
	ax1.imshow(xgrad)
	ax2.imshow(ygrad)
	ax3.imshow(edge)
	ax4.imshow(mask)
	plt.show()

###################################################################
"""
Functions for dataset analysis
"""
###################################################################

def calculate_dataset_ious(dataset):
	"""
	We want to return three items:
		(1) The number of instances where we have cell-cell overlap (ratio).
		(2) The degree of cell-cell overlap when there IS cell-cell overlap.
		(3) The degree of cell-cell overlap over the whole dataset.
	"""
	all_ious = []
	for im_ind in range(len(dataset.image_info)):
		M = dataset.load_mask(im_ind)
		#M should be an H x W x N array where N is the number of detections.
		#we will generate an N x N array for each image, where each element contains the IoU.
		this_ious = np.zeros(shape=(M.shape[2], M.shape[2]))
		#reducing repeating computations and self computation.
		#Array will not be square, but that's okay.
		for obj_a in range(M.shape[2]):
			for obj_b in range(obj_a+1, M.shape[2]):
				intersection = np.sum(np.logical_and(M[:,:,obj_a],M[:,:,obj_b]))
				union = np.sum(np.logical_or(M[:,:,obj_a],M[:,:,obj_b]))
				this_ious[obj_a,obj_b]=intersection/union
		all_ious.append(this_ious)
	#so now, calculate how many instances where we have cell to cell overlap.
	#this is anywhere is any of the iou matrices where there is a nonzero.
	#we also need to know the total number of overlap opportunities...
	opps = np.sum([np.sum(np.arange(1,M.shape[0])) for M in all_ious]) #for (3)
	did_overlap = np.sum([np.sum(M>0.) for M in all_ious]) #for (2) and (1).
	summed_ious = np.sum([np.sum(M) for M in all_ious])

	return did_overlap/opps, summed_ious/did_overlap, summed_ious/opps

###################################################################

def random_bright_color():
	import random
	import colorsys
	h, s, l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
	r, g, b = [int(256*i) for i in colorsys.hls_to_rgb(h,l,s)]
	return r,g,b

#add a script
