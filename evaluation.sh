CUDA_VISIBLE_DEVICES=0 \
python3 eval_voc.py \
--weight_path weight/181-best.pt \
--gpu_id 0 \
--visiual ./dsd \
--eval \
--mode val