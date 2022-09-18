README 
Training begin date and time: 20220513T112713
Notes: 5 slice 3D model at 20x.
CONFIGURATION SETTINGS: 
AVG_PIX                        None
Augmentors                     {'XYflip': True, 'Zflip': True, 'dense': True, 'stack': True, 'zoom': False, 'rotate': True, 'shear': False, 'blur': False, 'blur_out': False, 'brightness': False, 'blend': False, 'noise': False, 'contrast': False}
BATCH_SIZE                     8
EPOCHS                         50
GPU_COUNT                      1
GRADIENT_CLIP_NORM             5.0
GT_ASSIGN                      {'z_kmeans': True, 'pca_kmeans': False, 'slice_cells': False, 'slice_stack': False}
IMAGES_PER_GPU                 8
IMAGE_CHANNEL_COUNT            1
IMAGE_MAX_DIM                  256
IMAGE_MIN_DIM                  256
IMAGE_MIN_SCALE                0
IMAGE_RESIZE_MODE              centroid
INPUT_DIM                      3D
INPUT_IMAGE_SHAPE              [256 256  27   1]
INPUT_Z                        27
KERNEL_SIZE                    3
LEARNING_MOMENTUM              0.9
LEARNING_RATE                  0.02
LOSS_WEIGHTS                   {'CE_Loss_y2': 2.0, 'MSE_Loss_y0': 2.0, 'MSE_Loss_y1': 2.0, 'Dice_Loss_y3': 2.0}
NAME                           CellPose
NUM_CLASSES                    1
NUM_LAYER_FEATURES             [32, 64, 128, 256]
NUM_OUT                        4
OUT_CHANNELS                   5
PEN_opts                       {'collect': 'conv', 'kernels': [1, 3, 5, 7, 11], 'block_pool': 'conv', 'block_filters': 3}
Padding_Opts                   {'center': True, 'random': False}
STEPS_PER_EPOCH                50
TRAIN_BN                       False
UNET_DEPTH                     4
VALIDATION_STEPS               12
WEIGHT_DECAY                   1e-05
