CUDA_VISIBLE_DEVICES=0 python -i cell.py 'metrics' \
       --dataset="/home/physics/eddych/Environments/CellPose/Datasets/v7/val" \
       --weights="/home/physics/eddych/Environments/Mask_RCNN/mask_rcnn/logs/cell20220420T1553/mask_rcnn_cell_0049.h5" \
       --logs="/home/physics/eddych/Environments/Mask_RCNN/mask_rcnn/logs" #\
       #--subset='train'
			 #Mask_RCNN/mask_rcnn/datasets/v6" \
			 #"/home/physics/eddych/Environments/Mask_RCNN/mask_rcnn/logs/cell20201227T1143/mask_rcnn_cell_0072.h5" \
