"""
CellPose model architecture
"""
import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.backend as tfK
import tensorflow.keras.losses as tfKL
import tensorflow.keras.models as KM
import numpy as np
import os
import datetime
import tensorflow.keras as keras
import utils
import re
import multiprocessing

tf.compat.v1.disable_eager_execution()

################################################################################################
####################################INPUT HEAD LAYERS###########################################
################################################################################################

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
	x = KL.Lambda(lambda x: tfK.squeeze(x,-1),name=stem_squeeze_name)(x)
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
def stem_block(input_tensor, kernel_size, num_filters, z_slices,
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
	#t_shape = input_tensor.get_shape().as_list()
	#asser that the kernel size used must be less than the dimensions of the image
	assert any(x>=kernel_size for x in [z_slices] if x is not None)
	pad_size = kernel_size//2 #floor divide
	x = KL.ZeroPadding3D((pad_size,pad_size,0))(input_tensor)
	x = KL.Conv3D(num_filters if pool_mode=="conv" else 3, (kernel_size,kernel_size,kernel_size), name = conv_name)(x) #name=conv_name_base + '2a'
	x = KL.BatchNormalization(trainable=train_bn, name = bn_name)(x)
	x = KL.Activation('relu', name = act_name)(x)
	p_size = max(0, kernel_size - 1)
	#this is a question if batch size counts for t_shape...
	p_size = z_slices - p_size
	#p size is the pooling size of the z dimension.

	if pool_mode == "max":
		#this does maximum projection
		x = KL.MaxPooling3D(pool_size=(1, 1, p_size), strides=(1,1,1), name = pool_name+"_max")(x)
		#squeeze the z dimension, which is now down to one.
		x = KL.Lambda(lambda x: tfK.squeeze(x,-2),name=stem_squeeze_name)(x)
	#can we predict for sure, how large p_size is? well, tensor z is from config.INPUT+
	elif pool_mode == "conv":
		#could do a conv instead.
		x = KL.Conv3D(3, (1,1,p_size), name = pool_name+"_conv")(x)
		x = KL.BatchNormalization(trainable=train_bn, name = pool_name+"_bn")(x)
		x = KL.Activation('relu', name = pool_name+"_act")(x)
		#squeeze the z dimension, which is now down to one.
		x = KL.Lambda(lambda x: tfK.squeeze(x,-2),name=stem_squeeze_name)(x)
	elif pool_mode == "conv_split":
		#the previous conv does a convolution over the whole stack, learning F filters
		#from the entire stack which would act to separate R, G, B
		#Here, we use the previous convolution output to produce a single channel
		#image with 3 separate z-sections that encode position.
		#this operation is terribly slow compared to "conv". Why though?
		#rather than doing bn x (1024x1024x1) x 3 operations in conv, here we have to do
		# bn x (1024 x 1024 x 3)
		#frankly, it is fewer parameters to learn, and it seems just as many operations.
		k = int(np.ceil(p_size/3))
		s = int(np.floor(p_size/3))
		if s*2 + k < p_size:
			k += 1
		x = KL.Conv3D(1, kernel_size = (1,1,k), strides = (1, 1, s), name = pool_name+"_convsplit")(x)
		x = KL.BatchNormalization(trainable=train_bn, name = pool_name+"_bn")(x)
		x = KL.Activation('relu', name = pool_name+"_act")(x)
		#squeeze the z dimension, which is now down to one.
		x = KL.Lambda(lambda x: tfK.squeeze(x,-1),name=stem_squeeze_name)(x)
	else:
		print("Pooling option not available...")
		import pdb;pdb.set_trace()

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

	x = [stem_block(input_image, kernel_size=k, num_filters=PEN_opts['block_filters'], z_slices=z_slices, use_bias = True, train_bn=train_bn, pool_mode=PEN_opts['block_pool']) for k in kernels]

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
		x = KL.Lambda(lambda x: tfK.squeeze(x,-2),name="stem_collect_squeeze")(x)
		#check shape, should be B x H x W x 3
		#import pdb;pdb.set_trace()
	else:
		print("The designated config.PEN_opts['collect'] flag is not an option.")
		import pdb;pdb.set_trace()

	#this may help with training to make input into resnet normal
	#that is, every batch element will be normalized in range 0 - 1 (not channel-wise)
	x = KL.Lambda(lambda x: x - tf.reshape(tf.reduce_min(tf.reshape(x, [batch_size,-1]),axis=1),(batch_size,)+(1,)*3), name="stem_rescale_min")(x)
	x = KL.Lambda(lambda x: x / tf.reshape(tf.reduce_max(tf.reshape(x, [batch_size,-1]),axis=1),(batch_size,)+(1,)*3), name="stem_normalize")(x)
	#previously, commented here, it wasn't guarenteed to be in range 0-1.
	#x = KL.Lambda(lambda x: x / tf.reduce_max(tf.reshape(x,[-1,1]),axis=0) ,name="stem_normalize")(x)
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
	x_1 = KL.Lambda(lambda x: tfK.squeeze(x,-2),name='stem_squeeze')(x_1)
	#now go from [X,Y,1] to [X,Y,3] by copying to the other channels.
	x_1 = KL.Concatenate(axis=-1)([x_1,x_1,x_1])
	#stem_block(input_image, kernel_size=1,

	return x_1


################################################################################################
####################################DOWN PASS ##################################################
################################################################################################

##############DOWNPASS FUNCTIONS#############
def bnconv(input_tensor,k,f,dname,train_bn=True):
	conv_name = dname+'_bnconv_conv_k' + str(k) + '_f' + str(f)
	bn_name = dname+'_bnconv_bn_k' + str(k) + '_f' + str(f)
	act_name =  dname+'_bnconv_act_k' + str(k) + '_f' + str(f)
	"""
	Need to pass something for naming these layers.
	"""
	x = KL.BatchNormalization(trainable=train_bn,name=bn_name)(input_tensor)
	#x = KL.Activation('relu', name = act_name)(x)
	x = KL.Activation('relu',name=act_name)(x)
	#pad
	pad_size = int(np.floor(k/2))
	x = KL.ZeroPadding2D((pad_size,pad_size))(x)
	x = KL.Conv2D(f, (k,k),name=conv_name)(x) #strides 1. No
	return x

def identity_block(input_tensor,f,dname,train_bn=True):
	bn_name=dname+'_id_bn_f'+str(f)
	conv_name=dname+'_id_conv_f'+str(f)
	x = KL.BatchNormalization(trainable=train_bn,name=bn_name)(input_tensor)
	x = KL.Conv2D(f, (1,1), name=conv_name)(x)
	return(x)

def res_block(input_tensor, K, F, dname, train_bn=True):
	#K and F can be lists of kernels and corresponding number of filters to learn
	res_block_name = dname#+'_resb'
	for i,(k,f) in enumerate(zip(K,F)):
		if i==0:
			x = bnconv(input_tensor,k,f,train_bn=train_bn, dname=res_block_name+'_F'+str(i))
			#first F(x_t prime)
		else:
			x = bnconv(x,k,f,train_bn=train_bn,dname=res_block_name+'_F'+str(i))
			#second F(F(x_t prime))
	x2 = identity_block(input_tensor,f=F[-1],train_bn=train_bn, dname=res_block_name+'_P')
	#add x and x2
	x = tf.keras.layers.Add(name=res_block_name+'_add')([x, x2]) #returns tensor of the same shape as x and x2.
	#this returns x_t star.
	return x

def res_block_identity(input_tensor, K, F, dname, train_bn=True):
	#this is used in the second half of the downpass block
	res_block_name = dname#+'_resbi'
	for i,(k,f) in enumerate(zip(K,F)):
		if i==0:
			x = bnconv(input_tensor,k,f,train_bn=train_bn, dname=res_block_name+'_F'+str(i))
			#first F(x_t star)
		else:
			x = bnconv(x,k,f,train_bn=train_bn, dname=res_block_name+'_F'+str(i))
			#second F(F(x_t star))
	#this requires that the input tensor and x are the same shape. Used in second
	x = tf.keras.layers.Add(name=res_block_name+'_add')([x, input_tensor]) #returns tensor of the same shape as x and x2.
	#this returns x_t
	return x

def downsample_block(input_tensor):
	#max pool with kernel=(2,2), stride=(2,2)
	x = KL.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_tensor)
	return x

def make_style(input_tensor):
	"""
	okay so their code is as follows:
	where nn is from mxnet.gluon.nn
	style = nn.GlobalAvgPool2D()(x0)
	style = nn.Flatten()(style)
	style = F.broadcast_div(style , F.sum(style**2, axis=1).expand_dims(1)**.5)

	The mxnet nn global average pooling 2d takes a B x H x W x C tensor to B x C x 1 x 1, then flatten takes it to B x C
	Keras global average pooling goes from B x H x W x C to B x C, no need for a flatten

	broadcast_div is also mxnet, which simply divides each row in style by the sqrt(sum(squares)) of each row.
	In Keras, this is achieved via lambda?

	ISSUE WITH LAMBDA LAYER OPERATION.
	expand dims works fine in functional, since all it does is add a 1 to [None,etc]->[None,1,etc].
	sum, sqrt, and tile likely do not work as well.
	Need to find the work around. Well, this layer normalizes each batch, (sqrt sum squares) and maps to the same shape as x. Likely other functions exist for this.
	(1) see keras normalization layer
	(2) need to find. copy number to same shape as x.
	"""
	x = KL.GlobalAvgPool2D(name="Style")(input_tensor) #takes B x H x W x C array to B x C
	#There is an issue here. Initially, we don't know the batch size. That is, the input x here is shape [None, C].
	#we need this layer to be able to put up with this.
	#returns ValueError: Argument must be a dense tensor: (1, C) - got shape [2], but wanted [].
	#y = KL.Lambda(lambda x: tf.tile(tf.expand_dims(1/tfK.sqrt(tfK.sum(tfK.square(x),axis=1)),axis=-1),tf.constant(tf.expand_dims(x[0],axis=0).shape)))(x)
	#import pdb;pdb.set_trace()
	y = KL.Lambda(lambda x: tf.expand_dims(1/tfK.sqrt(tfK.sum(tfK.square(x),axis=1)),axis=-1))(x)
	#first argument does 1/sqrt(sum(squares)) and returns shape (B x 1)
	#multiplying by shape (B x C) automatically broadcasts the Bx1 into each C channel.
	#y = tf.tile(tf.expand_dims(1/tfK.sqrt(tfK.sum(tfK.square(x),axis=1)),axis=-1),tf.constant(tf.expand_dims(x[0],axis=0).shape))
	#this layer calculates each flattened rows 1/sqrt(sum(squares)), and then makes it into the same shape as x, for multiplication. That is
	# y = [A, A, A, A ...
	#	   B, B, B, B ...]
	#y should be shape B x C, same as x
	#multiply layer x and y
	x = KL.Multiply(name="Normalized_Style")([y, x]) #element wise multiplication #this casts the normalization appropriately.
	#performs normalization of the "style" for each batch example! Verified CE.
	return x

#######################DOWNPASS EXECUTIONS############################
def downpass_block(input_tensor, n, resK, resF, num_res=2, num_bnconv=2, train_bn=True):
	"""
	n = layer number
	resK = list of residual block kernels. Should be list of lists
		   i.e. [[3,3],[5,5]] will do a bnconv with k=3 then a subsequent bnconv with
		   k=3 in the first residual block, then a bnconv with k=5 then a subsequent bnconv with
		   k=5 in the second residual block. :: each list corresponds to a residual block.
		   Alternatively, [[3]] will do just a single bnconv with k=3 with a single residual block.
		   SHOULD BE a list of two lists only. Alternatively, could just pass one number.
	resF = list of residual block filters to learn. Should be a list of lists
		   follows same scheme as resK. Designates how many filters to learn in each bnconv step.
		   resF should be given every time. (nconv)

	resK = integer
	num_res = number (n) of res_block (1) + res_block_identity (n-1) to include.
	num_bnconv = number (n) of bnconv blocks to include in each residual block.
	#pool_size = down_pass pool
	CE: Added functionality so we can include more residual blocks (really, the second half of each downpass block)
		We can also change the kernels and number of filters learned by passing list of lists rather than int.
	"""
	downpass_block_name = "down"+str(n)

	if isinstance(resK,int):
		resK = [[resK]*num_bnconv]*num_res
	if isinstance(resF,int):
		resF = [[resF]*num_bnconv]*num_res
	#here, we want to pool
	if n>0:
		#pool.
		x = downsample_block(input_tensor)
	else:
		x = input_tensor
	assert len(resK)==2, "Kernel list is too large"
	assert len(resF)==2, "Filter list is too large"
	#the above assertions are only true if we are trying to code cellpose exactly.
	#I'm not sure this was supposed to be here.
	#x = res_block(x, resK[0], resF[0], train_bn=train_bn, dname=downpass_block_name+'_res0_conv')
	#x = res_block_identity(x, resK[1], resF[1], train_bn=train_bn, dname=downpass_block_name+'_res0_id')
	#For naming convention, please see literature for downpass.
	for i, (K,F) in enumerate(zip(resK,resF)):
		dname = downpass_block_name + '_res'+str(i)
		if i==0:
			x = res_block(x, K, F, train_bn=train_bn, dname=dname+'_xtstar') #so K here needs to be [3,3] and F should be at least [64,64]
		else:
			x = res_block_identity(x, K, F, train_bn=train_bn, dname=dname+'xt')
	return x

def build_downpass(input_tensor, N, K, nbase, train_bn=True):
	"""
	N = number of downpass layers to go through.
	K = kernel size for downpass convolutions (int or list)
		Could be passed as a list of lists
	nbase = Filter (channel) output for downpass (int or list)
		Could be passed as a list of lists
	"""
	assert isinstance(nbase,list), "F must be passed as a list"
	assert len(nbase)==N, "length of F must be equal to N"
	if isinstance(K,int):
		K = [K]*N
	assert len(K)==N, "length of K must be equal to N"
	xd = [] #we will store the downsample outputs.
	x = input_tensor
	for n in range(N):
		x = downpass_block(x, n, resK=K[n], resF=nbase[n], num_res=2, num_bnconv=2, train_bn=train_bn)
		xd.append(x)
	return xd

################################################################################################

################################################################################################
#################################### UP PASS ###################################################
################################################################################################

####################UP FUNCTIONS#########################

def bnconv_style(style, x, nconv, y=None, dname=None, concatenation=False, train_bn=True):
	"""
	Important note, the last bnconv in the literature suggests it should be BEFORE
	the addition of x and feat (style projection), but in their code, they do it AFTER.
	"""
	if dname is not None:
		bnconv_name = dname
	else:
		bnconv_name = 'style'
	#y is usually from the previous downpass layer to get concatenated
	if y is not None:
		if concatenation:
			print("concatenation code not yet programmed")
			pass
			#x = (concatenate code)
		else:
			x = KL.Add(name=bnconv_name+'_add_Fzprime_xt')([x,y])#not quite sure if these have the same sizes? they should.
	feat = KL.Dense(nconv,name=bnconv_name+'_ProjectS'+str(nconv))(style)# input should be batch_size x input_dim, behaves similar to mxnet gluon nn dense layer.
	#style from make_style is always batch_size x C features,
	#y = F.broadcast_add(x, feat.expand_dims(-1).expand_dims(-1))
	feat = KL.Lambda(lambda x: tf.expand_dims(tf.expand_dims(x,-2),-2), name=bnconv_name+'_reshape_style')(feat) #turns feat into a B x 1 x 1 x nconv
	#x = KL.Lambda(lambda x, y: tf.tile(y, tf.constant(tf.math.maximum(y.shape,x.shape))-tf.constant(y.shape)+1))([x, feat])
	#not working for some reason. TypeError: <lambda>() missing 1 required positional argument: 'y'
	# def temp(vec):
	# 	x, y = vec
	# 	return tf.tile(y, tf.constant(tf.math.maximum(y.shape,x.shape))-tf.constant(y.shape)+1)
	#x = KL.Lambda(temp)([x,feat])
	#import pdb; pdb.set_trace()
	#feat =  tf.tile(feat, tf.constant(tf.math.maximum(feat.shape,x.shape))-tf.constant(feat.shape)+1)
	#feat = KL.Lambda(lambda x: tf.tile(x[1], tf.constant(tf.math.maximum(x[1].shape,x[0].shape))-tf.constant(x[1].shape)+1))([x, feat])
	#THE ADD FUNCTION CASTS APPROPRIATELY AS WELL.
	#above works. Turns feat into same shape as x
	x = KL.Add(name=bnconv_name+'Add_style')([x,feat])
	x = bnconv(x, k=3, f=nconv, dname=bnconv_name+"_Fadd_x_projS", train_bn=train_bn) #See Github, my issue.
	#returns an output x that is the same shape as input x
	#x = bnconv(x, k=3, f=nconv) #double check this #returns an output that is b x H x W x nconv
	return x


def convup_block(input_tensor, prev_layer_tensor, style, nconv, dname, concatenation=False, train_bn=True):
	"""
	option for non-residual connection
	input_tensor is made the same shape as prev_layer_tensor with the first convolution.
	the upsampling should be done before convup.
	this layer comvines the tensor on the uppass with the style and the skip connection.
	"""
	add_name=dname + 'conv_'
	x = bnconv(input_tensor, k=3, f=nconv, dname=add_name+'F_zprime', train_bn=train_bn)
	x = bnconv_style(style, x, nconv=nconv, y=prev_layer_tensor, dname=add_name+'G1', concatenation=concatenation, train_bn=train_bn)
	return x

####################UP EXECUTIONS#######################

def resup_block(input_tensor, previous_layer_tensor, style, nconv, dname, concatenation=False, train_bn=True):
	#bnconv, bnconv_style, bnconv_style, bnconv_style
	#self.proj is idenity_block
	"""
	residual connection
	the first x,y take the input_tensor and turn it to the correct number of channels to be combined with the previous_layer tensor channels.
	"""
	add_name=dname + '_res'

	x = identity_block(input_tensor, nconv, dname=add_name+'0_zstar_P1', train_bn=train_bn)
	y = bnconv(input_tensor, k=3, f=nconv, dname=add_name+'0_zstar_F_zprime', train_bn=train_bn)
	y = bnconv_style(style, y, nconv, previous_layer_tensor, dname=add_name+'0_zstar_G1', concatenation=concatenation, train_bn=train_bn)
	x = KL.Add(name=add_name+'0_ztstar')([x,y])
	y = bnconv_style(style, x=x, nconv=nconv, dname=add_name+'1_G2', train_bn=train_bn)
	y = bnconv_style(style, x=y, nconv=nconv, dname=add_name+'1_G3', train_bn=train_bn)
	x = KL.Add(name=add_name+'1_zt')([y, x])
	return x


def build_uppass(style, xd, nbase, residual_on=True, concatenation=False, train_bn=True):
	"""
	xd = previous layer outputs (from build_downpass)
	nbase = list of nconv (filters) for each layer.
	so nbase should be the same list that gets passed to build_downpass, i.e. should be same as F
	should
	"""
	upname = 'up'
	if residual_on:
		x = resup_block(xd[-1], xd[-1], style, nbase[-1], dname=upname+str(len(nbase)), concatenation=concatenation, train_bn=train_bn)
	else:
		x = convup_block(xd[-1], xd[-1], style, nbase[-1], dname=upname+str(len(nbase)), concatenation=concatenation, train_bn=train_bn)

	for n in range(len(nbase)-2,-1,-1):
		#this takes the list in reverse. -2: -1 for the first layer, which is above, then -1 for the correct counting
		#ex, range(5,-1,-1) returns 5, 4, 3, 2, 1, 0.
		if residual_on:
			x = KL.UpSampling2D(size=(2,2),interpolation="nearest")(x)
			x = resup_block(x, xd[n], style, nbase[n], dname=upname+str(n), concatenation=concatenation, train_bn=train_bn)
		else:
			x = KL.UpSampling2D(size=(2,2),interpolation="nearest")(x)
			x = convup_block(x, xd[n], style, nbase[n], dname=upname+str(n), concatenation=concatenation, train_bn=train_bn)
	return x

################################################################################################
################################ CELLPOSE BACKBONE #############################################
################################################################################################
def cellpose_backbone_pass(x, N, K, nbase, style_on=True, train_bn=True):
	x = build_downpass(x, N, K, nbase, train_bn=train_bn)
	#make style
	style = make_style(x[-1]) #I'm not sure this is totally right. This makes the style out of the output of the last downpass.
	if not style_on:
		style=style*0.0
	#style should be a B x F[-1]
	x = build_uppass(style, xd=x, nbase=nbase, residual_on=True, concatenation=False, train_bn=train_bn)
	return x, style

################################################################################################
#################################### DEFINE LOSS ###############################################
################################################################################################
#Loss for map 3 (object detection)
def CE_loss(y_true,y_pred):
	# Cross entropy loss
	#BCE loss is more complicated than this. For every pixel, we have the ground truth 0-1
	#and the prediction from 0-1. Binary cross entropy expects a one-hot encoded representaiton.
	#that is, each pixel should be a 1xC where C is the number of classes.
	# so if the prediction is 0.6 and the ground truth is 1.0, then the loss is:
	#gt: [0,1.0]  pred: [0.4, 0.6]
	#sparse categorical cross entropy? Then future could prove additional labels.
	#output maps would all then need to be 1xC
	#y pred still needs to have form y_pred=[[P0, P1, P2...][P0,P1,P2]]
	return tfKL.SparseCategoricalCrossentropy()(y_true, tf.stack((1-y_pred,y_pred),axis=-1))

#Alternative for cross entropy, which has issue of class imbalance.
#Define focal loss.
def FL_loss(y_true, y_pred, gamma = 2., alpha = 0.25):
	"""
	BINARY FOCAL LOSS
	FL(p) = -alpha * (1-p)^gamma * log(p) if y_true==1
			 -(1-alpha) * p^gamma * log(1-p) if y_true==0
	https://medium.com/visionwizard/understanding-focal-loss-a-quick-read-b914422913e7
	y_pred = probability 0-1
	runs identical to tensorflow addons tfa.losses.SigmoidFocalCrossEntropy

	Although the loss value may be less compared to cross entropy, the gradient
	at probability p is significantly different from that of cross entropy.
	Notably, for negative cases (background) there is a smaller gradient for
	true negatives [easy cases] and a larger gradient for [false positives].
	For positive cases (object) there is a smaller gradient for true positives
	[easy cases] and a larger gradient for false negatives [hard cases]

	Tends to push predictions to around 0.5, rather than cross entropy which
	does well to push predictions to polarizing ends (0 and 1).
	Replacing dice loss with focal loss produces poor edges
	"""
	yhat = tfK.batch_flatten(tf.cast(y_true,tf.float32))
	p = tfK.batch_flatten(y_pred)
	return tfK.mean(-(1 - yhat + alpha*(2*yhat - 1)) * ((yhat + (1 - 2*yhat)*p)**gamma) * tfK.log(1 - yhat + (2*yhat - 1)*p + tfK.epsilon()))

def weighted_CE_loss(y_true, y_pred):
	#see model cellpose20220118T1233. Output prediction has poor definition of background vs object,
	#and poor resolution on cells, all likely due to over-emphasis on cell objects
	#https://rafayak.medium.com/how-do-tensorflow-and-keras-implement-binary-classification-and-the-binary-cross-entropy-function-e9413826da7
	#y_pred = probability 0-1
	#keras implementation is to first CLIP data
	y_pred = tfK.clip(y_pred, min_value = 10**-7, max_value = 0.9999999)
	#keras then converts the stable probabilities above to logits
	Z = tfK.log(y_pred / (1-y_pred) + tfK.epsilon())
	#finally, keras calculates cost(y, Z) = (1/m)*sum( max(Z, 0) * - Z*y + log(1 + exp(-abs(Z))))
	#we will do this momentarily.
	#first, calculate weight map.
	#shape of y_true is B x H x W x CH
	wmap = tf.expand_dims(tfK.max(tf.cast(y_true,tf.float32), axis=3), axis=-1) #shape of wmap is now B x H x W x 1
	#We need it to be float to do multiply and divide operation.
	wmap = tf.cast(wmap, tf.float32)
	#calculate the area of the max projected image.
	area = tf.cast(tf.shape(tfK.batch_flatten(wmap))[1],tf.float32) #total number of pixels (HxW)
	summed_obj = tfK.sum(tfK.batch_flatten(wmap)) #number of object pixels.
	rat = area / summed_obj #ratio of cell pixel weight to background. X : 1.
	rat = tf.expand_dims(tf.expand_dims(tf.expand_dims(rat,axis=-1),axis=-1),axis=-1) #shape is B x 1 x 1 x 1
	#multiply the per batch item maximum projected binary map (currently binary float) by the ratio
	wmap = rat * wmap #still B x H x W x 1
	wmap = tf.repeat(wmap, repeats = [tf.shape(y_true)[-1]], axis=-1) #shape is now B x H x W x CH
	#flatten for ease.
	wmap = tfK.batch_flatten(wmap)
	Z = tfK.batch_flatten(Z)
	y_true = tfK.batch_flatten(y_true)
	return tfK.mean(tfK.mean(wmap * (tf.math.maximum(Z,0.) - (Z * tf.cast(y_true,tf.float32)) + tfK.log(1 + tfK.exp(-tfK.abs(Z)) + tfK.epsilon())),axis=-1))


def MSE_loss(y_true,y_pred):
	#we want to unroll y_true and y_pred.
	#return tfKL.MeanSquaredError()(y_true,y_pred)
	#second mean is over batch
	return tfK.mean(tfK.mean(tfK.square(tfK.batch_flatten(y_true)-tfK.batch_flatten(y_pred)),axis=-1))

def Dice_loss(y_true,y_pred):
	#we want to unroll y_true and y_pred.
	#y_pred = tfK.abs(y_pred) > 0.5
	#I don't think we need to binarize this. Backprop won't work well with thresholding
	#y_pred = tfK.cast(tfK.abs(y_pred),'float32')
	y_true = tfK.cast(y_true,'float32')
	#y_true is mostly zeros, and therefore, those points will not contribute once multiplied.
	#this will teach it both edges and object from background.

	return 1. - tfK.mean((2. * tfK.sum(tfK.abs(tfK.batch_flatten(y_true) * tfK.batch_flatten(y_pred)),axis=-1) + 1e-6) / (tfK.sum(tfK.square(tfK.batch_flatten(y_true)),axis=-1) +  tfK.sum(tfK.square(tfK.batch_flatten(y_pred)),axis=-1) + 1e-6))


################################################################################################
#################################### BUILD MODEL ###############################################
################################################################################################

def build_model(input_shape, N, K, nbase, nout, out_channel, batch_size, z_slices, PEN_opts, style_on=True, train_bn=True, mode="training", input_dim="2D"):
	#do downpass.
	input_tensor = KL.Input(shape=(None,)*2 + tuple(list(input_shape[2:])))
	if input_dim=="2D":
		###CellPose Backbone####
		x, style = cellpose_backbone_pass(input_tensor, N, K, nbase, style_on=style_on, train_bn=train_bn)
	elif input_dim=="3D":
		#x = stem_graph_max_z(input_tensor, input_z=input_shape[2], train_bn=train_bn)
		###PEN###
		x = stem_graph_max(input_tensor, batch_size, z_slices, PEN_opts, train_bn=train_bn)
		###CellPose Backbone####
		x, style = cellpose_backbone_pass(x, N, K, nbase, style_on=style_on, train_bn=train_bn)
	#######HEAD########
	#return output to correct number of output units.
	x = bnconv(x, k=3, f=int(nout*out_channel), dname="conv_out", train_bn=train_bn) #make RGB output for each output
	d_in = KL.Activation('sigmoid',name="dice_out")(x[:,:,:,int(3*out_channel):int(4*out_channel)])

	l1_in = KL.Activation('sigmoid',name="act_out")(x[:,:,:,int(2*out_channel):int(3*out_channel)])
	#compute losses
	if mode=="training":
		gt_seg_map = KL.Input(shape=(None,)*3,name="gt_seg_map")#was input_shape[:2]
		#was (None,)*2. Changed to *3 for RGB channel.
		gt_x_grad = KL.Input(shape=(None,)*3,name="gt_x_grad")
		#flatten the grad.
		#gt_x = KL.Flatten()(gt_x_grad)
		gt_y_grad = KL.Input(shape=(None,)*3,name="gt_y_grad")
		#gt_y = KL.Flatten()(gt_y_grad)
		gt_borders = KL.Input(shape=(None,)*3,name="gt_borders")

		l1=KL.Lambda(lambda x: CE_loss(*x), name="CE_Loss_y2")(
			[gt_seg_map, l1_in]) #learn background vs object
		l2=KL.Lambda(lambda x: MSE_loss(*x), name="MSE_Loss_y0")(
			[gt_x_grad, x[:,:,:,:out_channel]]) #learn gradient
		l3=KL.Lambda(lambda x: MSE_loss(*x), name="MSE_Loss_y1")(
			[gt_y_grad, x[:,:,:,out_channel:int(2*out_channel)]]) #learn gradient
		l4=KL.Lambda(lambda x: Dice_loss(*x), name="Dice_Loss_y3")(
			[gt_borders, d_in]) #learn edgexs

		outputs = [x,style,l1,l2,l3,l4]
		inputs=[input_tensor, gt_seg_map, gt_x_grad, gt_y_grad, gt_borders]
	else:
		outputs = [l1_in, x[:,:,:,:out_channel], x[:,:,:,out_channel:int(2*out_channel)], d_in, style]
		inputs = [input_tensor]

	model = KM.Model(inputs,outputs,name="CPNet")
	return model


def build_stem_only(input_shape, batch_size, z_slices, PEN_opts, train_bn=False, input_dim="2D"):
	input_tensor = KL.Input(shape=(None,)*2 + tuple(list(input_shape[2:])))
	if input_dim=="3D":
		#x = stem_graph_max_z(input_tensor, input_z=input_shape[2], train_bn=train_bn)
		x = stem_graph_max(input_tensor, batch_size, z_slices, PEN_opts, train_bn=train_bn)
	else:
		x = input_tensor
	outputs = [x]
	inputs = [input_tensor]
	model = KM.Model(inputs,outputs,name="Stem")
	return model

################################################################################
###############################CELL POSE########################################
############################### NETWORK ########################################
################################################################################

class CPnet(object):
	"""Encapsulates the Cell Pose model functionality.
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
		#create log directories
		self.set_log_dir()
		#build network
		self.build()
		self.compiled=False

	def build(self):
		"""Build CellPose Model
			input_shape: The shape of the input image.
			mode: Either "training" or "inference". The inputs and
				outputs of the model differ accordingly.
		"""
		assert self.mode in ['training', 'inference']
		# Image size must be dividable by 2 multiple times
		h, w = self.config.INPUT_IMAGE_SHAPE[:2]
		if h / 2**(self.config.UNET_DEPTH) != int(h / 2**(self.config.UNET_DEPTH)) or w / 2**(self.config.UNET_DEPTH) != int(w / 2**(self.config.UNET_DEPTH)):
			raise Exception("Image size must be dividable by 2 at least {} times "
							"to avoid fractions when downscaling and upscaling."
							"For example, use 256, 320, 384, 448, 512, ... etc. ".format(self.config.UNET_DEPTH))

		print("Building network for {}...".format(self.mode))

		if hasattr(self.config,"INPUT_DIM"):
			#BUILD WITH STEM MODEL
			self.keras_model = build_model(input_shape = self.config.INPUT_IMAGE_SHAPE, \
												N = self.config.UNET_DEPTH, \
												K = self.config.KERNEL_SIZE, \
												nbase = self.config.NUM_LAYER_FEATURES, \
												nout = self.config.NUM_OUT, \
												batch_size = self.config.BATCH_SIZE if self.mode=="training" else 1, \
												z_slices = self.config.INPUT_Z, \
												out_channel = self.config.OUT_CHANNELS, \
												PEN_opts = self.config.PEN_opts, \
												style_on = True, \
												train_bn = True if self.mode=="training" else False, \
												mode = self.mode, \
												input_dim = self.config.INPUT_DIM)
		else:
			#BUILD WITHOUT STEM MODEL
			self.keras_model = build_model(input_shape = self.config.INPUT_IMAGE_SHAPE, \
												N = self.config.UNET_DEPTH, \
												K = self.config.KERNEL_SIZE, \
												nbase = self.config.NUM_LAYER_FEATURES, \
												nout = self.config.NUM_OUT, \
												batch_size = self.config.BATCH_SIZE if self.mode=="training" else 1, \
												z_slices = self.config.INPUT_Z, \
												out_channel = self.config.OUT_CHANNELS, \
												style_on = True, \
												train_bn = True if self.mode=="training" else False, \
												mode = self.mode)

		print("Complete.")


	def print_model(self, model_path = None):
		"""
		Use function to create a .png print out of your model.
		"""
		if model_path is not None:
			tf.keras.utils.plot_model(self.keras_model,to_file=model_path,show_layer_names=True,show_shapes=True)
		else:
			tf.keras.utils.plot_model(self.keras_model,to_file="cell_pose_keras.png",show_layer_names=True,show_shapes=True)
			print("Saved model graph under cell_pose_keras.png")

	def compile(self):
		"""Gets the model ready for training. Adds losses, regularization, and
		metrics. Then calls the Keras compile() function.
		"""
		# Optimizer object
		self.optimizer = keras.optimizers.SGD(
			lr=self.config.LEARNING_RATE, momentum=self.config.LEARNING_MOMENTUM,
			clipnorm=self.config.GRADIENT_CLIP_NORM)#, decay=0.0)
		# Add Losses and Metrics
		loss_names = [
			"CE_Loss_y2",  "MSE_Loss_y0", "MSE_Loss_y1", "Dice_Loss_y3"]
		for name in loss_names:
			layer = self.keras_model.get_layer(name)
			if layer.output in self.keras_model.losses:
				continue
			loss = (
				tf.reduce_mean(input_tensor=layer.output, keepdims=True)
				* self.config.LOSS_WEIGHTS.get(name, 1.))
			self.keras_model.add_loss(loss)
			#self.keras_model.metrics_names.append(name)
			#self.keras_model.add_metric(loss, name=name, aggregation='mean')

		# # Add L2 Regularization
		# Skip gamma and beta weights of batch normalization layers.
		reg_losses = [
			keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(input=w), tf.float32)
			for w in self.keras_model.trainable_weights
			if 'gamma' not in w.name and 'beta' not in w.name]
		self.keras_model.add_loss(tf.add_n(reg_losses))
		# self.keras_model.add_loss(lambda: tf.add_n([
		# 	keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(input=w), tf.float32)
		# 	for w in self.keras_model.trainable_weights
		# 	if 'gamma' not in w.name and 'beta' not in w.name]))
		#do we need to add a name for the L2 loss? I'm guessing that is what makes the val loss so HUGE.

		# Compile
		self.keras_model.compile(
			optimizer=self.optimizer,
			loss=[None] * len(self.keras_model.outputs))

		###THE ORDER HERE MATTERS.
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
		This function is useful to set certain layers to non-trainable
		"""
		# Print message on the first call (but not on recursive calls)
		if verbose > 0 and keras_model is None:
			log("Selecting layers to train...")

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

	# def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
	# 		  augmentation=False, custom_callbacks=None):
	# 	"""Train the model.
	# 	train_dataset, val_dataset: Training and validation Dataset objects.
	# 	learning_rate: The learning rate to train with
	# 	epochs: Number of training epochs. Note that previous training epochs
	# 			are considered to be done alreay, so this actually determines
	# 			the epochs to train in total rather than in this particaular
	# 			call.
	# 	layers: Allows selecting which layers to train. It can be:
	# 		- A regular expression to match layer names to train
	# 		- One of these predefined values:
	# 		  heads: The RPN, classifier and mask heads of the network
	# 		  all: All the layers
	# 		  stem: Train all Stem Layers
	# 		  - more options can be added per user's needs
	# 	augmentation: Optional boolean. An imgaug (https://github.com/aleju/imgaug)
	# 		augmentation.
	# 	custom_callbacks: Optional. Add custom callbacks to be called
	# 		with the keras fit_generator method. Must be list of type keras.callbacks.
	# 	"""
	# 	assert self.mode == "training", "Create model in training mode."
	# 	# Create log_dir if it does not exist
	# 	if not os.path.exists(self.log_dir):
	# 		os.makedirs(self.log_dir)
	#
	# 	# Callbacks
	# 	callbacks = [
	# 		keras.callbacks.TensorBoard(log_dir=self.log_dir,
	# 									histogram_freq=0, write_graph=True, write_images=False),
	# 		keras.callbacks.ModelCheckpoint(self.checkpoint_path,
	# 										verbose=0, save_weights_only=True, save_best_only=True,
	# 										monitor='val_loss',mode='min'),
	# 	]
	#
	# 	# Add custom callbacks to the list
	# 	if custom_callbacks:
	# 		callbacks += custom_callbacks
	#
	# 	#Pre-defined layer regular expressions
	# 	layer_regex = {
	# 		# All layers
	# 		"all": ".*",
	# 		# Just head layers (not yet in place)
	# 		"stem": r"(stem_conv.*)|(stem_bn.*)|(stem_act.*)",
	# 	}
	# 	if layers in layer_regex.keys():
	# 		layers = layer_regex[layers]
	#
	# 	# Train
	# 	log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
	# 	log("Checkpoint Path: {}".format(self.checkpoint_path))
	# 	#print("\nSetting layers as trainable...")
	# 	self.set_trainable(layers)
	# 	print("Compiling network losses...")
	# 	self.compile()
	# 	print("\nComplete. Beginning training...\n")
	#
	# 	# Data generators
	# 	train_generator = utils.data_generator(train_dataset, self.config, shuffle=True,
	# 									 augmentation=augmentation,
	# 									 batch_size=self.config.BATCH_SIZE)
	# 	val_generator = utils.data_generator(val_dataset, self.config, shuffle=True,
	# 								   batch_size=self.config.BATCH_SIZE)
	#
	# 	# Work-around for Windows: Keras fails on Windows when using
	# 	# multiprocessing workers. See discussion here:
	# 	# https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
	# 	if os.name == 'nt':
	# 		workers = 0
	# 	else:
	# 		workers = multiprocessing.cpu_count()
	#
	# 	# XXX make multiprocessing work again
	# 	workers = 1
	#
	# 	steps_per_epoch_train=int(np.ceil(len(train_dataset.image_info)/self.config.BATCH_SIZE))#how big is the train set?
	# 	steps_per_epoch_val = int(np.ceil(len(val_dataset.image_info)/self.config.BATCH_SIZE))
	#
	# 	#import pdb;pdb.set_trace()
	#
	# 	# self.keras_model.fit(
	# 	# 	train_generator,
	# 	# 	initial_epoch=self.epoch,
	# 	# 	epochs=epochs,
	# 	# 	steps_per_epoch=self.config.STEPS_PER_EPOCH,
	# 	# 	callbacks=callbacks,
	# 	# 	validation_data=val_generator,
	# 	# 	validation_steps=self.config.VALIDATION_STEPS,
	# 	# 	max_queue_size=100,
	# 	# 	workers=workers,
	# 	# 	use_multiprocessing=workers > 1,
		# # )
		# self.keras_model.fit(
		# 	train_generator,
		# 	initial_epoch=self.epoch,
		# 	epochs=epochs,
		# 	steps_per_epoch=steps_per_epoch_train,
		# 	callbacks=callbacks,
		# 	validation_data=val_generator,
		# 	validation_steps=steps_per_epoch_val,
		# 	max_queue_size=100,
		# 	workers=workers,
		# 	use_multiprocessing=workers > 1,
	# 	)
	# 	self.epoch = max(self.epoch, epochs)
	#
	#

	def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
			  augmentation=False, custom_callbacks=None, train_it=None):
		"""Train the model.
		train_dataset, val_dataset: Training and validation Dataset objects.
		learning_rate: The learning rate to train with
		epochs: Number of training epochs. Note that previous training epochs
				are considered to be done alreay, so this actually determines
				the epochs to train in total rather than in this particaular
				call.
		layers: Allows selecting which layers to train. It can be:
			- A regular expression to match layer names to train
			- One of these predefined values:
			  all: All the layers
			  stem: Train all Stem Layers
			  body: Train all layers EXCEPT PEN.
			  - more options can be added per user's needs
		augmentation: Optional boolean. An imgaug (https://github.com/aleju/imgaug)
			augmentation.
		custom_callbacks: Optional. Add custom callbacks to be called
			with the keras fit_generator method. Must be list of type keras.callbacks.
		"""
		assert self.mode == "training", "Create model in training mode."
		self.compile_train_params(train_dataset, val_dataset, learning_rate, epochs, layers,
				  augmentation=augmentation, custom_callbacks=custom_callbacks, train_it=train_it)
		# )
		train_generator = utils.data_generator(self.train_dataset, self.config, shuffle=True,
										 augmentation=augmentation,
										 batch_size=self.config.BATCH_SIZE)
		#added CE 11/24/21
		#need to make sure validation is reproducible, which also involves removing steps_per_epoch_val in fit call.
		# def_setting = self.config.IMAGE_RESIZE_MODE
		# self.config.IMAGE_RESIZE_MODE = "center" #setting this to center without any augmentation
		# #makes the validation set reproducible. But let's imagine that, well,
		# #I don't care.
		# val_generator = utils.data_generator(self.val_dataset, self.config, shuffle=True,
		# 							   batch_size=self.config.BATCH_SIZE)
		val_generator = utils.data_generator(self.val_dataset, self.config, shuffle=True,
									   augmentation=augmentation,
									   batch_size=self.config.BATCH_SIZE)
		#self.config.IMAGE_RESIZE_MODE = def_setting
		self.config.write_config_txt(self.log_dir)

		#import pdb;pdb.set_trace()
		self.keras_model.fit(
			train_generator,
			initial_epoch=self.epoch,
			epochs=epochs,
			steps_per_epoch=self.config.STEPS_PER_EPOCH,
			callbacks=self.callbacks,
			validation_data=val_generator,
			validation_steps=self.steps_per_epoch_val,
			max_queue_size=100,
			workers=self.workers,
			use_multiprocessing=self.workers > 1,
			verbose=1)


	def compile_train_params(self, train_dataset, val_dataset, learning_rate, epochs, layers,
			  augmentation=False, custom_callbacks=None, train_it=None):
		"""Train the model. Compile all parameters
		Run this prior to training. It should
		train_dataset, val_dataset: Training and validation Dataset objects.
		learning_rate: The learning rate to train with
		epochs: Number of training epochs. Note that previous training epochs
				are considered to be done alreay, so this actually determines
				the epochs to train in total rather than in this particaular
				call.
		layers: Allows selecting which layers to train. It can be:
			- A regular expression to match layer names to train
			- One of these predefined values:
			  heads: The RPN, classifier and mask heads of the network
			  all: All the layers
			  stem: Train all Stem Layers
			  body: Train all layers EXCEPT PEN.
			  - more options can be added per user's needs
		augmentation: Optional boolean. An imgaug (https://github.com/aleju/imgaug)
			augmentation.
		custom_callbacks: Optional. Add custom callbacks to be called
			with the keras fit_generator method. Must be list of type keras.callbacks.
		"""
		if not self.compiled:
			print("\nCompiling training parameters...\n")
			assert self.mode == "training", "Create model in training mode."
			# Create log_dir if it does not exist
			if not os.path.exists(self.log_dir):
				os.makedirs(self.log_dir)

			# Callbacks
			self.callbacks = [
				keras.callbacks.TensorBoard(log_dir=self.log_dir,
											histogram_freq=0, write_graph=True, write_images=False),
				keras.callbacks.ModelCheckpoint(self.checkpoint_path,
												verbose=0, save_weights_only=True, save_best_only=True,
												monitor='val_loss',mode='min'),
			]

			# Add custom callbacks to the list
			if custom_callbacks:
				self.callbacks += custom_callbacks

			#Pre-defined layer regular expressions
			layer_regex = {
				# All layers
				"all": ".*",
				# Just head layers (not yet in place)
				"stem": r"(stem_conv.*)|(stem_bn.*)|(stem_pool.*)|(stem_act.*)",
				#Layers except stem.
				"body": r"^(?!.*(stem_conv|stem_bn|stem_pool|stem_act)).*",
			}

			if layers in layer_regex.keys():
				if layers=="all":
					print("TRAINING ALL LAYERS.")
				elif layers=="stem":
					print("TRAINING PEN NETWORK ONLY.")
				elif layers=="body":
					print("TRAINING CELLPOSE NETWORK LAYERS ONLY.")
				layers = layer_regex[layers]

			self.learning_rate = learning_rate

			# Train
			log("\nStarting at epoch {}. LR={}\n".format(self.epoch, self.learning_rate))
			log("Checkpoint Path: {}".format(self.checkpoint_path))
			#print("\nSetting layers as trainable...")
			self.set_trainable(layers)
			print("Compiling network losses...")
			self.compile()
			print("\nComplete. Beginning training...\n")

			# Data generators
			self.train_dataset = train_dataset
			self.augmentation = augmentation
			self.val_dataset = val_dataset

			# Work-around for Windows: Keras fails on Windows when using
			# multiprocessing workers. See discussion here:
			# https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
			if os.name == 'nt':
				self.workers = 0
			else:
				self.workers = multiprocessing.cpu_count()

			# XXX make multiprocessing work again
			self.workers = 1

		# if train_it is not None:
		# 	self.steps_per_epoch_train = int(train_it)
		# else:
		# 	self.steps_per_epoch_train = int(np.ceil(len(train_dataset.image_info)/self.config.BATCH_SIZE))
		# 	#int(np.ceil(len(train_dataset.image_info)/self.config.BATCH_SIZE))#how big is the train set?
		self.steps_per_epoch_val = len(val_dataset.image_info)//self.config.BATCH_SIZE#int(np.ceil(len(val_dataset.image_info)/self.config.BATCH_SIZE))

		self.compiled=True


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
		if self.config.NAME is not None:
			self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
				self.config.NAME.lower(), now))
			self.checkpoint_path = os.path.join(self.log_dir, "{}_*epoch*.h5".format(
				self.config.NAME.lower()))
		else:
			self.log_dir = os.path.join(self.model_dir, "{:%Y%m%dT%H%M}".format(now))
			self.checkpoint_path = os.path.join(self.log_dir, "cellpose_*epoch*.h5")



		# Path to save after each epoch. Include placeholders that get filled by Keras.
		self.checkpoint_path = self.checkpoint_path.replace(
			"*epoch*", "{epoch:04d}")


class StemNet(object):
	"""Encapsulates the stem portion model functionality.
	The actual Keras model is in the keras_model property.
	"""

	def __init__(self, config):
		"""
		mode: Either "training" or "inference"
		config: A Sub-class of the Config class
		model_dir: Directory to save training logs and trained weights
		"""
		self.config = config
		self.compiled = False
		#build network
		self.build()

	def build(self):
		"""Build stem Model
			input_shape: The shape of the input image.
		"""
		print("Building stem network...")

		self.keras_model = build_stem_only(self.config.INPUT_IMAGE_SHAPE, \
											batch_size = 1, \
											z_slices = self.config.INPUT_Z, \
											PEN_opts = self.config.PEN_opts, \
											train_bn = False, \
											input_dim=self.config.INPUT_DIM)

		print("Complete.")

	def print_model(self, model_path = None):
		"""
		Use function to create a .png print out of your model.
		"""
		if model_path is not None:
			tf.keras.utils.plot_model(self.keras_model,to_file=model_path,show_layer_names=True,show_shapes=True)
		else:
			tf.keras.utils.plot_model(self.keras_model,to_file="stem_model_keras.png",show_layer_names=True,show_shapes=True)
			print("Saved model graph under stem_model_keras.png")


##############################################################
######################LOAD WEIGHTS FUNCTION###################
##############################################################

def load_weights(model, weights_path, exclude=None):
	"""
	model = keras api model. Requires matching layer names to saved model weights (self.keras_model)
	weights_path = path to .h5 file containing model weights
	exclude = list of layer names to exclude from loading weights.
	ex. ["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"]
	"""
	# Select weights file to load
	import h5py
	if h5py is None:
		raise ImportError('"load_weights" requires h5py.')

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
		model.load_weights(weights_path, by_name=True)

	for layer,initial in zip(model.layers,initial_weights):
		weights = layer.get_weights()
		if weights and all(tf.nest.map_structure(np.array_equal,weights,initial)):
			print(f'Checkpoint contained no weights for layer {layer.name}!')
	print("done loading weights")

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
