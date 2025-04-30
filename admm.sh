casename="ramen"
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
    # render rgb
    # python render.py -m output_admm/${casename}_${level}
    # render language features
    CUDA_VISIBLE_DEVICES=$((level+2)) nohup python render.py \
        -m output_admm/${casename}_${level} \
        --include_feature \
        --ckpt ${iter} \
        > logs/${casename}_render_lvl${level}.log 2>&1 &
    # e.g. python render.py -m output/sofa_3 --include_feature
done

wait  # 等待所有后台任务完成

cd eval
nohup sh eval_admm.sh > logs/${casename}_eval_lvl${level}.log 2>&1 &