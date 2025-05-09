#!/bin/bash

# 按顺序的 每个scene都正常训练然后评估
#casenames=("bed" "bench" "blue_sofa" "covered_desk" "lawn" "office_desk" "room" "snacks" "sofa" "table")
casenames=("covered_desk" "snacks" "table")

mkdir -p logs/langsplat/train_ae
mkdir -p logs/langsplat/render_ae
mkdir -p logs/langsplat/train
mkdir -p logs/langsplat/render
mkdir -p logs/langsplat/eval

# ---------- 阶段 1：训练 + 渲染 ----------
for casename in "${casenames[@]}"
do
    echo "Processing ${casename}"

    # ---------- 训练autoencoder ----------
    
        cd autoencoder

        CUDA_VISIBLE_DEVICES=3 nohup python train.py \
            --dataset_path ../data/3dovs/${casename} \
            --dataset_name ${casename} \
            --encoder_dims 256 128 64 32 3 \
            --decoder_dims 16 32 64 128 256 256 512 \
            --lr 0.0007 \
            > ../logs/langsplat/train_ae/${casename}.log 2>&1 &
        wait 

        CUDA_VISIBLE_DEVICES=3 nohup python test.py \
            --dataset_path ../data/3dovs/${casename} \
            --dataset_name ${casename} \
            > ../logs/langsplat/render_ae/${casename}.log 2>&1 &

        wait
    

    # ---------- train langsplat ----------
    cd ..
    for level in 1 2 3
    do
        CUDA_VISIBLE_DEVICES=$((level+2)) nohup python train.py \
            -s data/3dovs/${casename} \
            -m output/${casename} \
            --start_checkpoint data/3dovs/${casename}/output/${casename}/chkpnt30000.pth \
            --port 600${level} \
            --feature_level ${level} \
            > logs/langsplat/train/${casename}_lv${level}.log 2>&1 &
    done
    wait  # 等待训练完成

    # ---------- render langsplat ----------
    for level in 1 2 3
    do
        CUDA_VISIBLE_DEVICES=$((level+2)) nohup python render.py \
            -m output/${casename}_${level} \
            --include_feature \
            > logs/langsplat/render/${casename}_lv${level}.log 2>&1 &
    done
    wait  # 等待渲染完成


    CUDA_VISIBLE_DEVICES=3 bash -c "
        cd eval
        nohup sh eval_3dovs.sh ${casename} > ../logs/langsplat/eval/${casename}.log 2>&1
    " &
done


wait  # 等待所有评估完成
echo "所有任务完成"
