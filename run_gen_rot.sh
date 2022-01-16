#!/bin/sh

python data/generate_rotations.py --outfile data/pert_060.csv --deg 60 --format wt \
--dataset-path data/ModelNet40 --categoryfile data/categories/modelnet40_half1.txt #\
# --resume result_train/result_0_model.pth --logfile ./result_train/result_0.log