#!/bin/sh

python train.py -data modelnet --outfile ./result_train/resampling \
--pretrained none -j 12 \
--tf_cfg data/transforms/resampling.yaml --param_cfg data/params/modelnet40/train.yaml #\
# --resume result_train/result_0_model.pth --logfile ./result_train/result_0.log