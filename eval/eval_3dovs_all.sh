#!/bin/bash

CUDA_VISIBLE_DEVICES=0 nohup  sh eval_3dovs.sh bed   > _bed.log   2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup  sh eval_3dovs.sh bench > _bench.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup  sh eval_3dovs.sh lawn  > _lawn.log  2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup  sh eval_3dovs.sh room  > _room.log  2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup  sh eval_3dovs.sh sofa  > _sofa.log  2>&1 &

echo "All eval jobs have been started in background with nohup."
