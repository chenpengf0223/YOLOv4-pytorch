CUDA_VISIBLE_DEVICES=0 \
nohup python3 -u train.py \
    --resume \
    --weight_path weight/last.pt \
    --gpu_id 0 > nohup.log 2>&1 &