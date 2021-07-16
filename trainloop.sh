#!/bin/sh

cd /home/layout_tf/Mask_RCNN/Layout

python layout.py train --dataset='/home/Dataset'   --weights='coco' --layers='heads' --lr=0.001 --epochs=25 \
&& python layout.py train --dataset='/home/Dataset'   --weights="last" --layers='heads' --lr=0.0001 --epochs=40 \
&& python layout.py train --dataset='/home/Dataset' --weights="last" --layers='4+' --lr=0.0001 --epochs=60  \
&& python layout.py train --dataset='/home/Dataset' --weights="last" --layers='all' --lr=0.0001 --epochs=100
