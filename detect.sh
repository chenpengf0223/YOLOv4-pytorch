CUDA_VISIBLE_DEVICES=0 \
python3 eval_voc.py \
    --weight_path weight/best.pt \
    --gpu_id 0 \
    --visiual /home/chenp/YOLOv4-pytorch/qixing/qifeng \
    --eval --mode det