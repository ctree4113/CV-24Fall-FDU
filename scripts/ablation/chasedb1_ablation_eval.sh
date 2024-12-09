#! /bin/sh
cd ../..

# Evaluate Baseline
python eval.py \
--dataset 'chase' \
--data_root '/root/autodl-tmp/datas/CHASEDB1' \
--resize 960 960 \
--metrics clDice:DSC:IOU:1-Betti \
--num-class 1 \
--model_type 'base' \
--ckpt_path '/root/autodl-tmp/ablation/CHASEDB1/baseline' \
--output_path '/root/autodl-tmp/eval_results/ablation/CHASEDB1/baseline' \
--num_pred 5

# Evaluate MRDE Only
python eval.py \
--dataset 'chase' \
--data_root '/root/autodl-tmp/datas/CHASEDB1' \
--resize 960 960 \
--metrics clDice:DSC:IOU:1-Betti \
--num-class 1 \
--model_type 'improved' \
--use_mrde \
--ckpt_path '/root/autodl-tmp/ablation/CHASEDB1/mrde_only' \
--output_path '/root/autodl-tmp/eval_results/ablation/CHASEDB1/mrde_only' \
--num_pred 5

# Evaluate GLFI Only
python eval.py \
--dataset 'chase' \
--data_root '/root/autodl-tmp/datas/CHASEDB1' \
--resize 960 960 \
--metrics clDice:DSC:IOU:1-Betti \
--num-class 1 \
--model_type 'improved' \
--use_glfi \
--ckpt_path '/root/autodl-tmp/ablation/CHASEDB1/glfi_only' \
--output_path '/root/autodl-tmp/eval_results/ablation/CHASEDB1/glfi_only' \
--num_pred 5

# Evaluate Full Model
python eval.py \
--dataset 'chase' \
--data_root '/root/autodl-tmp/datas/CHASEDB1' \
--resize 960 960 \
--metrics clDice:DSC:IOU:1-Betti \
--num-class 1 \
--model_type 'improved' \
--use_mrde \
--use_glfi \
--freq_weight 0.1 \
--topo_weight 0.1 \
--ckpt_path '/root/autodl-tmp/ablation/CHASEDB1/full_model' \
--output_path '/root/autodl-tmp/eval_results/ablation/CHASEDB1/full_model' \
--num_pred 5 