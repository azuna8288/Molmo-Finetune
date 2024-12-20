export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTHONPATH=src:$PYTHONPATH
export MODEL_NAME="Molmo-7B-D-0924"
modelscope download --model "LLM-Research/Molmo-7B-D-0924" --local_dir $MODEL_NAME
export OUT_DIR="output/lora_fp32_epoch3_bs1_ga4"
deepspeed --master_port 1237 src/training/train.py \
    --lora_enable True \
    --lora_rank 128 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --deepspeed scripts/zero2.json \
    --model_id $MODEL_NAME \
    --data_path data/chunk01.json,data/chunk02.json,data/chunk02_enhanced.json,data/chunk01_enhanced.json \
    --image_folder data/images \
    --freeze_vision_tower False \
    --freeze_llm False \
    --tune_projector True \
    --bf16 False \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir $OUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --projector_lr 1e-5 \
    --vision_lr 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.02 \
    --lr_scheduler_type "cosine" \
    --logging_steps 100 \
    --tf32 True \
    --gradient_checkpointing False \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 16
    
cp $OUT_DIR/model*  $MODEL_NAME/
python3 upload_model.py