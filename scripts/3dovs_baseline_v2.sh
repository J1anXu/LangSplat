#!/bin/bash

# 按顺序的 每个scene都正常训练然后评估
casenames=("bed" "bench" "blue_sofa" "covered_desk" "lawn" "office_desk" "room" "snacks" "sofa" "table")
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

done


# ---------- 阶段 2：并行评估 ----------
echo "开始评估阶段..."
eval_gpu_list=(0 1 2 3 4 5)
max_jobs=${#eval_gpu_list[@]}
pids=()

for i in "${!casenames[@]}"
do
    casename=${casenames[$i]}
    gpu_index=$((i % max_jobs))
    gpu_id=${eval_gpu_list[$gpu_index]}

    echo "开始评估 ${casename} 使用 GPU ${gpu_id}"

    # 启动评估任务
    CUDA_VISIBLE_DEVICES=$gpu_id bash -c "
        cd eval
        nohup sh eval_3dovs.sh ${casename} > ../logs/langsplat/eval/${casename}.log 2>&1
    " &

    pids+=($!)

    # 控制并发数量：最多同时 $max_jobs 个任务
    if (( ${#pids[@]} >= max_jobs )); then
        wait -n  # 等待任意一个完成
        # 移除已完成的 PID
        new_pids=()
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=($pid)
            fi
        done
        pids=("${new_pids[@]}")
    fi
done

# 等待所有剩余任务完成
wait
echo "所有评估任务完成"


