#!/bin/bash

# 专门用来复现的脚本,循环N次,每次都把pth保存到一个新的地方

casenames=("sofa")
N=1

for baseline_idx in $(seq 1 $N); do

    RANDOM_STR=$(cat /dev/urandom | tr -dc a-z0-9 | head -c 4)
    echo "Random string: $RANDOM_STR"
    mkdir -p output/${RANDOM_STR}

    for casename in "${casenames[@]}"; do
        echo "开始处理 ${casename}"

        # ---------- 训练 ----------
        for level in 1 2 3; do
            CUDA_VISIBLE_DEVICES=$((level+2)) nohup python train.py \
                -s data/3dovs/${casename} \
                -m output/${RANDOM_STR}/${casename} \
                --start_checkpoint data/3dovs/${casename}/output/${casename}/chkpnt30000.pth \
                --port 600${level} \
                --feature_level ${level} \
                --baseline_idx ${baseline_idx} \
                > output/${RANDOM_STR}/${casename}_train_lv_${level}.log 2>&1 &
        done

        wait  

        # ---------- 渲染 ----------
        for level in 1 2 3; do
            CUDA_VISIBLE_DEVICES=$((level+2)) nohup python render.py \
                -m output/${RANDOM_STR}/${casename}_${level} \
                --include_feature \
                > output/${RANDOM_STR}/${casename}_render_lv_${level}.log 2>&1 &
        done

        wait  

        python eval/evaluate_iou_loc2.py \
                --dataset_name ${casenames} \
                --feat_dir output/${RANDOM_STR} \
                --ae_ckpt_dir autoencoder/ckpt \
                --output_dir  output/${RANDOM_STR}/eval_result \
                --mask_thresh 0.4 \
                --encoder_dims 256 128 64 32 3 \
                --decoder_dims 16 32 64 128 256 256 512 \
                --dataset_path "data/3dovs/"  
    done

done
