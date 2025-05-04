#!/bin/bash

# 数据集路径
DATASET_PATH=../data/3dovs

# 场景名称列表
SCENES=("bed" "bench" "blue_sofa" "covered_desk" "lawn" "office_desk" "room" "snacks" "sofa" "table")

# 可用 GPU 列表
GPUS=(0 1 2 3 4 5 6)


# 创建日志目录（如果不存在）
LOG_DIR=logs
mkdir -p $LOG_DIR

# 获取总场景数和GPU数
NUM_SCENES=${#SCENES[@]}
NUM_GPUS=${#GPUS[@]}

# 初始化索引
i=0

while [ $i -lt $NUM_SCENES ]; do
    for gpu_index in "${!GPUS[@]}"; do
        if [ $i -ge $NUM_SCENES ]; then
            break
        fi

        SCENE_NAME=${SCENES[$i]}
        GPU_ID=${GPUS[$gpu_index]}
        LOG_FILE=$LOG_DIR/${SCENE_NAME}_test.log

        echo "Launching training for scene: $SCENE_NAME on GPU $GPU_ID (log: $LOG_FILE)"

        CUDA_VISIBLE_DEVICES=$GPU_ID \
        nohup python test.py \
            --dataset_path $DATASET_PATH/$SCENE_NAME \
            --dataset_name $SCENE_NAME > $LOG_FILE 2>&1 &

        ((i++))
    done

    # 等待当前批次任务完成再进入下一轮
    wait
    echo "Finished one batch of training. Proceeding to next..."
done

echo "All training processes completed."
