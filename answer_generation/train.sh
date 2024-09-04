export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_ENABLE_MPS_FALLBACK=1

python train.py \
    --pretrained_model "uer/t5-base-chinese-cluecorpussmall" \
    --save_dir "checkpoints/DuReaderQG" \
    --train_path "data/DuReaderQG/train.json" \
    --dev_path "data/DuReaderQG/dev.json" \
    --img_log_dir "logs/DuReaderQG" \
    --img_log_name "T5-Base-Chinese" \
    --batch_size 4 \
    --max_source_seq_len 20 \
    --max_target_seq_len 8 \
    --learning_rate 5e-5 \
    --num_train_epochs 30 \
    --logging_steps 100 \
    --valid_steps 50 \
    --device "mps:0"

#mps:0
#cpu
#cuda:0