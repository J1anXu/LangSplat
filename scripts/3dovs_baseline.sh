#!/bin/bash

#casenames=("figurines" "ramen" "teatime" "waldo_kitchen")
casenames=("bed" "bench" "blue_sofa" "covered_desk" "lawn" "office_desk" "room" "snacks" "sofa" "table")
mkdir -p logs/langsplat/train
mkdir -p logs/langsplat/render
mkdir -p logs/langsplat/eval

# ---------- 阶段 1：训练 + 渲染 ----------
for casename in "${casenames[@]}"
do
    echo "开始处理 ${casename}"

    # ---------- 训练阶段 ----------
    for level in 1 2 3
    do
        CUDA_VISIBLE_DEVICES=$((level+2)) nohup python train.py \
            -s data/3dovs/${casename} \
            -m output/${casename} \
            --start_checkpoint data/3dovs/${casename}/output/${casename}/chkpnt30000.pth \
            --port 600${level} \
            --feature_level ${level} \
            > logs/langsplat/train/${casename}_train_lvl${level}.log 2>&1 &
    done

    wait  # 等待训练完成

    # ---------- 渲染阶段 ----------
    for level in 1 2 3
    do
        CUDA_VISIBLE_DEVICES=$((level+2)) nohup python render.py \
            -m output/${casename}_${level} \
            --include_feature \
            > logs/langsplat/render/${casename}_render_lvl${level}.log 2>&1 &
    done

    wait  # 等待渲染完成
done

# ---------- 阶段 2：并行评估 ----------
eval_gpu_list=(2 3 4 5)
eval_job_idx=0

for casename in "${casenames[@]}"
do
    eval_gpu=${eval_gpu_list[$((eval_job_idx % ${#eval_gpu_list[@]}))]}
    echo "开始评估 ${casename} 使用 GPU ${eval_gpu}"

    CUDA_VISIBLE_DEVICES=$eval_gpu bash -c "
        cd eval
        nohup sh eval.sh ${casename} > ../logs/langsplat/eval/${casename}_eval.log 2>&1
    " &

    ((eval_job_idx++))
done

wait  # 等待所有评估完成
echo "所有任务完成"
