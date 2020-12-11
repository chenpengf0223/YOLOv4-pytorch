CUDA_VISIBLE_DEVICES=0 \
nohup python3 -u train.py \
    --weight_path weight/mobilenetv3.pth \
    --gpu_id 0 > nohup.log 2>&1 &