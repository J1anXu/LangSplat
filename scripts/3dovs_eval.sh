#!/bin/bash

mkdir -p logs/langsplat/train_ae
mkdir -p logs/langsplat/render_ae
mkdir -p logs/langsplat/train
mkdir -p logs/langsplat/render
mkdir -p logs/langsplat/eval

CUDA_VISIBLE_DEVICES=0 bash -c "
    cd eval
    nohup sh eval_3dovs.sh bed > ../logs/langsplat/eval/bed.log 2>&1
"&

CUDA_VISIBLE_DEVICES=1 bash -c "
    cd eval
    nohup sh eval_3dovs.sh bench > ../logs/langsplat/eval/bench.log 2>&1
"&

CUDA_VISIBLE_DEVICES=2 bash -c "
    cd eval
    nohup sh eval_3dovs.sh room > ../logs/langsplat/eval/room.log 2>&1
"&

CUDA_VISIBLE_DEVICES=3 bash -c "
    cd eval
    nohup sh eval_3dovs.sh sofa > ../logs/langsplat/eval/sofa.log 2>&1
"&

CUDA_VISIBLE_DEVICES=4 bash -c "
    cd eval
    nohup sh eval_3dovs.sh lawn > ../logs/langsplat/eval/lawn.log 2>&1
"&
