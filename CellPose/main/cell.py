import argparse
import os
import sys
#import python files
import model as modellib
import config as configurations
import utils as utils
import copy
import datetime
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback, TerminateOnNaN



def train(model, dataset, dataset_dir, data_type):
	"""Train the model."""
	if data_type=="MouseVision":
		import textwrap
		#MouseVision is not automatically split into two. In this case, lets set
		#the random seed for reproduceability. Secondly, split the dataset into two.
		dataset_all=copy.deepcopy(dataset)
		print("Loading Dataset...")
		dataset_all.load_cell(dataset_dir)
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
		dataset_train=copy.deepcopy(dataset)
		dataset_train.load_cell(dataset_dir, "train")#, model.config.INPUT_DIM)
		dataset_train.prepare()

		# Validation dataset
		print("Loading the validation set")
		dataset_val=copy.deepcopy(dataset)
		dataset_val.load_cell(dataset_dir, "val")
		dataset_val.prepare()

	augmentation = True

	print("TRAINING ALL LAYERS for {} epochs".format(config.EPOCHS))

	print("\n Warning: A large MSE will result at first, as the network begins to find the range for the gradients.")

	rlrop = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10)
	stop_on_nan = TerminateOnNaN()
	print_rl = MyCallback()
	reg_loss = RegLossCallback()
	#import pdb;pdb.set_trace()
	#layers was 'all'
	model.train(dataset_train, dataset_val,
				learning_rate=config.LEARNING_RATE,
				epochs=int(config.EPOCHS),
				augmentation=augmentation,
				layers='body',
				custom_callbacks=[rlrop, print_rl, stop_on_nan, reg_loss])

def MouseTrain(model, dataset, dataset_dir, data_type):
	"""Train the model."""
	if data_type=="MouseVision":
		import textwrap
		#MouseVision is not automatically split into two. In this case, lets set
		#the random seed for reproduceability. Secondly, split the dataset into two.
		dataset_all=copy.deepcopy(dataset)
		print("Loading Dataset...")
		dataset_all.load_cell(dataset_dir)
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
		dataset_train=copy.deepcopy(dataset)
		dataset_train.load_cell(dataset_dir, "train")#, model.config.INPUT_DIM)
		dataset_train.prepare()

		# Validation dataset
		print("Loading the validation set")
		dataset_val=copy.deepcopy(dataset)
		dataset_val.load_cell(dataset_dir, "val")
		dataset_val.prepare()

	augmentation = True

	#return dataset_train, dataset_val

	rlrop = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10)
	stop_on_nan = TerminateOnNaN()
	print_rl = MyCallback()

	print("TRAINING ALL LAYERS for {} epochs".format(config.EPOCHS))

	print("\n Warning: A large MSE will result at first, as the network begins to find the range for the gradients.")

	print("\nBeginning training, with lr at 0.2 and 50 iterations for numerical stability for one epoch...\n")

	#import pdb;pdb.set_trace()
	model.train(dataset_train, dataset_val,
				learning_rate=config.LEARNING_RATE,
				epochs=1,
				augmentation=augmentation,
				layers='all',
				train_it=50,
				custom_callbacks=[rlrop, print_rl, stop_on_nan])

	print("\nBeginning training, annealing learning rate from 0 to 0.2 over ten epochs...")
	for lr in np.linspace(0,config.LEARNING_RATE,10):
		#reset learning rate
		model.keras_model.optimizer.learning_rate.assign(lr)
		print(model.keras_model.optimizer.learning_rate)
		#tfKL.set_value(model.keras_model.optimizer.learning_rate, 0.2)

		model.train(dataset_train, dataset_val,
					learning_rate=lr,
					epochs=1,
					augmentation=augmentation,
					layers='all',
					custom_callbacks=[print_rl, stop_on_nan])

	print("\nTraining, with constant learning rate of 0.2 for 390 epochs...")
	model.keras_model.optimizer.learning_rate.assign(0.2)
	model.train(dataset_train, dataset_val,
				learning_rate=config.LEARNING_RATE,
				epochs=390,
				augmentation=augmentation,
				layers='all',
				custom_callbacks=[rlrop, print_rl, stop_on_nan])

	print("\nFinal training, with learning rate reduced by factor of two every 10 epochs for 100 epochs...")
	for lr in np.array([config.LEARNING_RATE]*10)/np.array([2,4,8,16,32,64,128,256,512,1024]):
		model.keras_model.optimizer.learning_rate.assign(lr)
		model.train(dataset_train, dataset_val,
					learning_rate=lr,
					epochs=10,
					augmentation=augmentation,
					layers='all',
					custom_callbacks=[print_rl, stop_on_nan])


def detect(model, config, dataset, dataset_dir, save_dir):
	dataset_test = copy.deepcopy(dataset)
	dataset_test.load_cell(dataset_dir)
	dataset_test.prepare()
	for image_id,_ in enumerate(dataset_test.image_info):
		#load image.
		print("Detecting on image: {}".format(os.path.join(dataset_test.image_info[image_id]['path'],dataset_test.image_info[image_id]['id'])))
		image,window = utils.load_image_inference(dataset_test, config, image_id)
		hmin,hmax,wmin,wmax=window
		#need to add dimension for batch
		image = np.expand_dims(image, axis=0)
		#likely also need to pad probably.
		#run detection
		#
		"""
		This may work the first time through prediction, but after it is possible
		that the network will expect an image of the same shape!
		"""
		[mask, xgrad, ygrad, style]=model.predict([image],verbose=0)
		#import pdb;pdb.set_trace()
		image = image[0,hmin:hmax,wmin:wmax,:]
		#lets also save a max projection of the image.
		image = np.max(image,axis=2)
		mask = mask[0,hmin:hmax,wmin:wmax]
		xgrad = xgrad[0,hmin:hmax,wmin:wmax]
		ygrad = ygrad[0,hmin:hmax,wmin:wmax]
		#import pdb;pdb.set_trace()
		np.savez_compressed(os.path.join(save_dir,dataset_test.image_info[image_id]['id']),image=image, mask=mask, xgrad=xgrad, ygrad=ygrad)

##############################################################
##################VISUALIZNG STEM PERFORMANCE#################
##############################################################
def create_stem_model(config, weights_path):
	"""
	Weights should already been loaded
	"""
	model = modellib.StemNet(config)

	#if given, load weights.
	if weights_path:
		modellib.load_weights(model.keras_model, weights_path)
	return model

def visualize_stem_out(model, dataset, image_id=0):
	"""
	Visualize the output from stem (or just input) into network
	Using dataset, will analyze image idx.
	"""
	print("Visualizing stem output on image: {}".format(os.path.join(dataset.image_info[image_id]['path'],dataset.image_info[image_id]['id'])))
	image,window = utils.load_image_inference(dataset, config, image_id)
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

################################################################
################################################################
################CALL BACKS #####################################
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


#
# Root directory of the project
ROOT_DIR = os.path.abspath("./")

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
#if not os.path.isdir(DEFAULT_LOGS_DIR):
#    raise ImportError("'DEFAULT_LOGS_DIR' does not point to an exisiting directory.")
# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# Parse command line arguments
parser = argparse.ArgumentParser(
	description='Cell-Pose for cell segmentation',formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--mode", required=True,
					metavar="<command>",
					help="'training' or 'inference'")
parser.add_argument('--dataset', required=True,
					metavar="/path/to/dataset",
					help='Root directory of the dataset. \nTraining MUST have form:\n\
dataset_directory\n\
----/train\n\
--------/images\n\
--------/gt\n\
----/val\n\
--------/images\n\
--------/gt\n\
Inference MUST have form:\n\
dataset_directory\n\
----/images')
parser.add_argument('--weights', required=False,
					metavar="/path/to/weights.h5",
					help="Path to weights .h5 file if you wish to load previously trained weights.")
parser.add_argument('--logs', required=False,
					default=DEFAULT_LOGS_DIR,
					metavar="/path/to/logs/",
					help='Logs and checkpoints directory (default=logs/)')
parser.add_argument('--subset', required=False,
					metavar="Dataset sub-directory",
					help="Subset of dataset to run prediction on. \
					ex. 'test' will only pull data from dataset/test")
parser.add_argument('--data_type', required=True,
					metavar="MouseVision, MouseVisionSplit, Cell3D, Cell2D",
					help="Specifies which type of configuration and model to load. \
					Options: 'MouseVision', 'MouseVisionSplit', 'Cell3D', 'Cell2D'")
args = parser.parse_args()

# Validate arguments
assert args.mode in ["training", "inference", "visualize"], "mode argument must be one of 'training' or 'inference' or 'visualize'"
assert args.data_type in ["MouseVision", "MouseVisionSplit", "Cell3D", "Cell2D"], "data_type argument must be one of 'MouseVision', 'MouseVisionSplit', 'Cell3D', 'Cell2D'"
if args.dataset[-1]=="/":
	args.dataset=args.dataset[0:-1]

if args.mode=="visualize":
	assert args.weights, "--weights argument must given (path to weights) for visualization of input."
# if args.mode == "training":
# 	assert args.dataset, "Argument --dataset is required for training"
#elif args.command == "detect":
#    assert args.subset, "Provide --subset to run prediction on"
############################################################################
if args.weights:
	print("Weights: ", args.weights)
if args.dataset:
	print("Dataset: ", args.dataset)
if args.subset:
	print("Subset: ", args.subset)
print("Logs: ", args.logs)
print("Data from: ", args.data_type)

#Create appropriate configuration and model.
if args.data_type=="MouseVision":
	config = configurations.Config()
	dataset=utils.CellPoseDataset()
elif args.data_type=="MouseVisionSplit":
	config = configurations.Config()
	dataset=utils.CellPoseDataset()
elif args.data_type=="Cell3D":
	config = configurations.Cell3DConfig()
	dataset=utils.CellDataset()
elif args.data_type=="Cell2D":
	config = configurations.Cell2DConfig()
	dataset = utils.CellDataset()

if args.mode=="inference":
	#we do not want to reshape our images.
	config.IMAGE_RESIZE_MODE='none'

config.display()


#If training, train
if args.mode=="training":
	#create model and build network (training or inference)
	#here, the model does not have a pre-specified shape (meaning the expected input
	#is None x None x None x C or None x None x None x Z x 1) making it more flexible
	model = modellib.CPnet(args.mode, config, args.logs)

	#if given, load weights.
	if args.weights:
		modellib.load_weights(model.keras_model, args.weights)

	if args.data_type=="MouseVision" or args.data_type=="MouseVisionSplit":
		MouseTrain(model, dataset, args.dataset, args.data_type)
	else:
		train(model, dataset, args.dataset, args.data_type)
	#dataset.load_cell(args.dataset)
elif args.mode=="inference":
	#create model and build network (training or inference)
	#here, the model does not have a pre-specified shape (meaning the expected input
	#is None x None x None x C or None x None x None x Z x 1) making it more flexible
	model = modellib.CPnet(args.mode, config, args.logs)

	#if given, load weights.
	if args.weights:
		modellib.load_weights(model.keras_model, args.weights)

	# Create directory
	if not os.path.exists(RESULTS_DIR):
		os.makedirs(RESULTS_DIR)
	submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
	submit_dir = os.path.join(RESULTS_DIR, submit_dir)
	os.makedirs(submit_dir)
	#import pdb;pdb.set_trace()
	detect(model.keras_model, config, dataset, args.dataset, submit_dir)

elif args.mode=="visualize":
	import tifffile
	print("Prepping dataset...")
	dataset_test = copy.deepcopy(dataset)
	dataset_test.load_cell(args.dataset)
	dataset_test.prepare()
	print("Complete.")
	if hasattr(config, 'INPUT_DIM'):
		if config.INPUT_DIM=="3D":
			model = create_stem_model(config, args.weights)
			print("Printing Stem Model layers...")
			model.keras_model.summary()
			IM, IM_stem = visualize_stem_out(model, dataset_test, image_id=0)
		else:
			print("\nModel is a 2D input model, therefore there is no stem to visualize.")
			print("Instead, returning input image.\n")
			IM, IM_stem = visualize_stem_out(model=None, dataset=dataset_test, image_id=0)
	else:
		print("\nModel is a 2D input model, therefore there is no stem to visualize.")
		print("Instead, returning input image.\n")
		IM, IM_stem = visualize_stem_out(model=None, dataset=dataset_test, image_id=0)

	tifffile.imwrite(os.path.join(os.path.dirname(args.weights),dataset_test.image_reference(0)+'_StemOut.tif'),IM_stem,photometric='rgb')
	tifffile.imwrite(os.path.join(os.path.dirname(args.weights),dataset_test.image_reference(0)+'_StemOut_R.tif'),IM_stem[:,:,0])
	tifffile.imwrite(os.path.join(os.path.dirname(args.weights),dataset_test.image_reference(0)+'_StemOut_G.tif'),IM_stem[:,:,1])
	tifffile.imwrite(os.path.join(os.path.dirname(args.weights),dataset_test.image_reference(0)+'_StemOut_B.tif'),IM_stem[:,:,2])
	import matplotlib.pyplot as plt
	fig, (ax1,ax2,ax3)=plt.subplots(1,3)
	ax1.imshow(IM_stem[:,:,0])
	ax2.imshow(IM_stem[:,:,1])
	ax3.imshow(IM_stem[:,:,2])
	plt.show()

"""
Ltot=0
Llist = np.zeros(50)
L1tot=0
L1list = np.zeros(50)
L2tot=0
L2list = np.zeros(50)
L3tot=0
L3list = np.zeros(50)

#import matplotlib.pyplot as plt
train_generator = utils.data_generator(dataset_train, config, shuffle=True, augmentation=True, batch_size=8)
val_generator = utils.data_generator(dataset_val, config, shuffle=True,  batch_size=8)
for g in range(50):
	print(g)
	inputs,outputs = next(val_generator)
	[L,l1,l2,l3]=model.keras_model.evaluate(inputs)
	Ltot+=L
	L1tot+=l1
	L2tot+=l2
	L3tot+=l3
	Llist[g]=L
	L1list[g]=l1
	L2list[g]=l2
	L3list[g]=l3
IM,M,X,Y=inputs
n=2
fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2)
ax1.imshow(IM[n])
ax2.imshow(M[n])
ax3.imshow(X[n])
ax4.imshow(Y[n])
plt.show()
"""
