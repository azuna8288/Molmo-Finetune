## 设置代理
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118 
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118 
export no_proxy=.http://byted.org


export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTHONPATH=src:$PYTHONPATH
export MODEL_NAME="./Molmo-7B-D-0924"
modelscope download --model "LLM-Research/Molmo-7B-D-0924" --local_dir $MODEL_NAME
OUT_DIR="output/fp32_epoch3_bs1_ga4"
deepspeed --master_port 1237 --include="localhost:0,1,2,3,4,5,6,7"  src/training/train.py \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path data/ground_train/chunk01.json,data/ground_train/chunk02.json,data/ground_train/chunk02_enhanced.json,data/ground_train/chunk01_enhanced.json \
    --image_folder data/images \
    --freeze_vision_tower True \
    --freeze_llm False \
    --tune_projector True \
    --bf16 False \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir $OUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --projector_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
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