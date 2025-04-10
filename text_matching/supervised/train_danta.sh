export HF_ENDPOINT=https://hf-mirror.com
#export PYTORCH_ENABLE_MPS_FALLBACK=1

python train_pointwise.py \
    --model "nghuyong/ernie-3.0-base-zh" \
    --train_path "data/comment_classify/train.txt" \
    --dev_path "data/comment_classify/dev.txt" \
    --save_dir "checkpoints/comment_classify" \
    --img_log_dir "logs/comment_classify" \
    --img_log_name "ERNIE-PointWise" \
    --batch_size 8 \
    --max_seq_len 128 \
    --valid_steps 50 \
    --logging_steps 10 \
    --num_train_epochs 10 \
    --device "mps:0"
#  --device "cuda:0"