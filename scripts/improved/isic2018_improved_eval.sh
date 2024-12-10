#! /bin/sh
cd ../..

python eval.py \
--dataset 'isic' \
--data_root '/data/agent/Jetbrains/test/CV-24Fall-FDU/data/ISIC2018_npy_all_224_320' \
--resize 224 320 \
--metrics DSC:IOU:ACC:PREC \
--num-class 1 \
--decoder_attention \
--use_glfi \
--use_mrde \
--ckpt_path /data/agent/Jetbrains/test/CV-24Fall-FDU/ckpt/improved/ISIC2018 \
--output_path /data/agent/Jetbrains/test/CV-24Fall-FDU/eval_results/improved/ISIC2018 \
--num_pred 5

