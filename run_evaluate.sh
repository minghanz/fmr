#!/bin/sh
# python evaluate.py -data modelnet --uniformsampling True --mag 360 --resampling --outfile ./result/result_360_resampling.csv
# python evaluate.py -data modelnet --uniformsampling True --mag 80 --resampling --outfile ./result/result_80_resampling.csv
# python evaluate.py -data modelnet --uniformsampling True --mag 70 --resampling --outfile ./result/result_70_resampling.csv
# python evaluate.py -data modelnet --uniformsampling True --mag 60 --resampling --outfile ./result/result_60_resampling.csv
# python evaluate.py -data modelnet --uniformsampling True --mag 50 --resampling --outfile ./result/result_50_resampling.csv
# python evaluate.py -data modelnet --uniformsampling True --mag 40 --resampling --outfile ./result/result_40_resampling.csv
# python evaluate.py -data modelnet --uniformsampling True --mag 30 --resampling --outfile ./result/result_30_resampling.csv
# python evaluate.py -data modelnet --uniformsampling True --mag 20 --resampling --outfile ./result/result_20_resampling.csv
# python evaluate.py -data modelnet --uniformsampling True --mag 10 --resampling --outfile ./result/result_10_resampling.csv
# python evaluate.py -data modelnet --uniformsampling True --mag 00 --resampling --outfile ./result/result_00_resampling.csv

# python evaluate.py -data modelnet --uniformsampling True --mag 360 --noise --outfile ./result/result_360_noise.csv
# python evaluate.py -data modelnet --uniformsampling True --mag 80 --noise --outfile ./result/result_80_noise.csv
# python evaluate.py -data modelnet --uniformsampling True --mag 70 --noise --outfile ./result/result_70_noise.csv
# python evaluate.py -data modelnet --uniformsampling True --mag 60 --noise --outfile ./result/result_60_noise.csv
# python evaluate.py -data modelnet --uniformsampling True --mag 50 --noise --outfile ./result/result_50_noise.csv
# python evaluate.py -data modelnet --uniformsampling True --mag 40 --noise --outfile ./result/result_40_noise.csv
# python evaluate.py -data modelnet --uniformsampling True --mag 30 --noise --outfile ./result/result_30_noise.csv
# python evaluate.py -data modelnet --uniformsampling True --mag 20 --noise --outfile ./result/result_20_noise.csv
# python evaluate.py -data modelnet --uniformsampling True --mag 10 --noise --outfile ./result/result_10_noise.csv
# python evaluate.py -data modelnet --uniformsampling True --mag 00 --noise --outfile ./result/result_00_noise.csv

# python evaluate.py -data modelnet --uniformsampling True --mag 90 --noise --outfile ./result/result_90_noise.csv
# python evaluate.py -data modelnet --uniformsampling True --mag 110 --noise --outfile ./result/result_110_noise.csv
# python evaluate.py -data modelnet --uniformsampling True --mag 130 --noise --outfile ./result/result_130_noise.csv
# python evaluate.py -data modelnet --uniformsampling True --mag 150 --noise --outfile ./result/result_150_noise.csv
# python evaluate.py -data modelnet --uniformsampling True --mag 170 --noise --outfile ./result/result_170_noise.csv
# python evaluate.py -data modelnet --resampling --mag 30 --noise --outfile ./result/result_30_resampling_test.csv

python evaluate.py -data modelnet --outfile ./result_val/ideal_resampling_60 \
--pretrained result_train/ideal_model.pth --tf_cfg data/transforms/resampling.yaml --param_cfg data/params/modelnet40/val.yaml \
--perturbations data/pert_060.csv

# python evaluate.py -data modelnet --mag 00 --outfile ./result/result_00_dup.csv
# python evaluate.py -data modelnet --outfile ./result/result_ttest_30_dup_test_unf_dbnoi_comrot_0t \
# --pretrained result/fmr_model_modelnet40.pth --tf_cfg data/transforms/ideal.yaml --param_cfg data/conditions/modelnet40/val.yaml
# python evaluate.py -data modelnet --mag 60 --outfile ./result/result_60_dup.csv
# python evaluate.py -data modelnet --mag 90 --outfile ./result/result_90_dup.csv
# python evaluate.py -data modelnet --mag 120 --outfile ./result/result_120_dup.csv
# python evaluate.py -data modelnet --mag 150 --outfile ./result/result_150_dup.csv
# python evaluate.py -data modelnet --mag 180 --outfile ./result/result_180_dup.csv

# python evaluate.py -data modelnet --mag 30 --density --outfile ./result/result_30_density.csv --device cpu
# python evaluate.py -data modelnet --mag 90 --density --outfile ./result/result_90_density.csv --device cpu
# python evaluate.py -data modelnet --mag 180 --density --outfile ./result/result_180_density.csv --device cpu
# python evaluate.py -data modelnet --mag 00 --density --outfile ./result/result_00_density.csv --device cpu
# python evaluate.py -data modelnet --mag 60 --density --outfile ./result/result_60_density.csv --device cpu
# python evaluate.py -data modelnet --mag 120 --density --outfile ./result/result_120_density.csv --device cpu
# python evaluate.py -data modelnet --mag 150 --density --outfile ./result/result_150_density.csv --device cpu

# python evaluate.py -data 7scene --mag 0 --mag_trans 0 --duo-mode --outfile ./result_7scene/result_00_duo_mode.csv
# python evaluate.py -data 7scene --mag 30 --mag_trans 0 --duo-mode --outfile ./result_7scene/result_30_duo_mode.csv
# python evaluate.py -data 7scene --mag 60 --mag_trans 0 --duo-mode --outfile ./result_7scene/result_60_duo_mode.csv
# python evaluate.py -data 7scene --mag 90 --mag_trans 0 --duo-mode --outfile ./result_7scene/result_90_duo_mode.csv
# python evaluate.py -data 7scene --mag 120 --mag_trans 0 --duo-mode --outfile ./result_7scene/result_120_duo_mode.csv
# python evaluate.py -data 7scene --mag 150 --mag_trans 0 --duo-mode --outfile ./result_7scene/result_150_duo_mode.csv
# python evaluate.py -data 7scene --mag 180 --mag_trans 0 --duo-mode --outfile ./result_7scene/result_180_duo_mode.csv
# python evaluate.py -data 7scene --mag 45 --mag_trans 0 --duo-mode --outfile ./result_7scene/result_45_duo_mode.csv


# python evaluate.py -data modelnet --outfile ./result/result_ovn_bench_2.csv --cfg data/cfg_onetdata.yaml --pretrained result/fmr_model_modelnet40.pth #--device cpu