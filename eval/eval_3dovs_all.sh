CUDA_VISIBLE_DEVICES=0 sh eval_3dovs.sh bed &
CUDA_VISIBLE_DEVICES=1 sh eval_3dovs.sh bench &
CUDA_VISIBLE_DEVICES=2 sh eval_3dovs.sh lawn &
CUDA_VISIBLE_DEVICES=3 sh eval_3dovs.sh room &
CUDA_VISIBLE_DEVICES=4 sh eval_3dovs.sh sofa &
