#!/bin/bash
#casenames=("figurines" "ramen" "teatime" "waldo_kitchen")
casenames=("figurines")
iter=2500
# rm -rf logs/admm
# rm -rf output_admm
mkdir -p logs/admm/train
mkdir -p logs/admm/render
mkdir -p logs/admm/eval

# ---------- 阶段 1：训练和渲染 ----------
for casename in "${casenames[@]}"
do
    echo "开始处理 ${casename}"

    # 训练阶段
    for level in 1 2 3
    do
        CUDA_VISIBLE_DEVICES=$((level+2)) nohup python train_admm.py \
            -s data/lerf_ovs/${casename} \
            -m output_admm/${casename} \
            --start_checkpoint output/${casename}_${level}/chkpnt30000.pth \
            --port 600${level} \
            --feature_level ${level} \
            > logs/admm/train/${casename}_train_lvl${level}.log 2>&1 &
    done

    wait  # 等待训练完成

    # 渲染阶段
    for level in 1 2 3
    do
        CUDA_VISIBLE_DEVICES=$((level+2)) nohup python render.py \
            -m output_admm/${casename}_${level} \
            --include_feature \
            --ckpt ${iter} \
            > logs/admm/render/${casename}_render_lvl${level}.log 2>&1 &
    done

    wait  # 等待渲染完成
done

# ---------- 阶段 2：评估 ----------
eval_gpu_list=(3 4 5 6)
eval_job_idx=0

for casename in "${casenames[@]}"
do
    eval_gpu=${eval_gpu_list[$((eval_job_idx % ${#eval_gpu_list[@]}))]}
    echo "开始评估 ${casename} 使用 GPU ${eval_gpu}"

    CUDA_VISIBLE_DEVICES=$eval_gpu bash -c "
        cd eval
        nohup sh eval_admm.sh ${casename} > ../logs/admm/eval/${casename}_eval.log 2>&1
    " &

    ((eval_job_idx++))
done

wait  # 等待所有评估完成
echo "所有任务完成"
