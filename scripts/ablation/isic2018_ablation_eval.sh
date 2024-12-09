#! /bin/sh
cd ../..

# Evaluate Baseline
python eval.py \
--dataset 'isic' \
--data_root '/root/autodl-tmp/datas/ISIC2018_npy_all_224_320' \
--resize 224 320 \
--metrics DSC:IOU:ACC:PREC \
--num-class 1 \
--model_type 'base' \
--ckpt_path '/root/autodl-tmp/ablation/ISIC2018/baseline' \
--output_path '/root/autodl-tmp/eval_results/ablation/ISIC2018/baseline' \
--num_pred 5

# Evaluate MRDE Only
python eval.py \
--dataset 'isic' \
--data_root '/root/autodl-tmp/datas/ISIC2018_npy_all_224_320' \
--resize 224 320 \
--metrics DSC:IOU:ACC:PREC \
--num-class 1 \
--model_type 'improved' \
--use_mrde \
--ckpt_path '/root/autodl-tmp/ablation/ISIC2018/mrde_only' \
--output_path '/root/autodl-tmp/eval_results/ablation/ISIC2018/mrde_only' \
--num_pred 5

# Evaluate GLFI Only
python eval.py \
--dataset 'isic' \
--data_root '/root/autodl-tmp/datas/ISIC2018_npy_all_224_320' \
--resize 224 320 \
--metrics DSC:IOU:ACC:PREC \
--num-class 1 \
--model_type 'improved' \
--use_glfi \
--ckpt_path '/root/autodl-tmp/ablation/ISIC2018/glfi_only' \
--output_path '/root/autodl-tmp/eval_results/ablation/ISIC2018/glfi_only' \
--num_pred 5

# Evaluate Full Model
python eval.py \
--dataset 'isic' \
--data_root '/root/autodl-tmp/datas/ISIC2018_npy_all_224_320' \
--resize 224 320 \
--metrics DSC:IOU:ACC:PREC \
--num-class 1 \
--model_type 'improved' \
--use_mrde \
--use_glfi \
--freq_weight 0.1 \
--topo_weight 0.1 \
--ckpt_path '/root/autodl-tmp/ablation/ISIC2018/full_model' \
--output_path '/root/autodl-tmp/eval_results/ablation/ISIC2018/full_model' \
--num_pred 5 