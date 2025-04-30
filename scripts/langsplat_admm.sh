casename="waldo_kitchen"
iter=10000
for level in 1 2 3

do
    CUDA_VISIBLE_DEVICES=$((level+2)) nohup python train_admm.py \
        -s data/lerf_ovs/${casename} \
        -m output_admm/${casename} \
        --start_checkpoint output/${casename}_${level}/chkpnt30000.pth \
        --port 600${level} \
        --feature_level ${level} \
        > logs/${casename}_train_lvl${level}.log 2>&1 &
done

wait  # 等待所有后台任务完成

for level in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$((level+2)) nohup python render.py \
        -m output_admm/${casename}_${level} \
        --include_feature \
        --ckpt ${iter} \
        > logs/${casename}_render_lvl${level}.log 2>&1 &
done

wait  # 等待所有后台任务完成

cd eval
nohup sh eval_admm.sh ${casename}> ../logs/${casename}_eval.log 2>&1 &
