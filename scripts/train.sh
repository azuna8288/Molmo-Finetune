export MODEL_NAME="./Molmo-7B-D-0924"
modelscope download --model "LLM-Research/Molmo-7B-D-0924" --local_dir $MODEL_NAME

# Training parameters
learning_rate="1e-5"
num_train_epochs="3"
per_device_train_batch_size=1
gradient_accumulation_steps=4
projector_lr="1e-5"
vision_lr="1e-5"
freeze_vision_tower="False"
freeze_llm="False"
tune_projector="True"


OUT_DIR="output/epoch${num_train_epochs}_bs${per_device_train_batch_size}_ga${gradient_accumulation_steps}_lr${learning_rate}_proj_lr${projector_lr}_vision_lr${vision_lr}_freeze_vision_tower${freeze_vision_tower}_freeze_llm${freeze_llm}_tune_projector${tune_projector}"

deepspeed --master_port 1237 --include="localhost:0,1,2,3,4,5,6,7"  src/training/train.py \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path data/grounding/ground_train/chunk01.json,data/grounding/ground_train/chunk02.json,data/grounding/ground_train/chunk02_enhanced.json,data/grounding/ground_train/chunk01_enhanced.json \
    --image_folder data/images \
    --freeze_vision_tower $freeze_vision_tower \
    --freeze_llm $freeze_llm \
    --tune_projector $tune_projector \
    --bf16 False \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir $OUT_DIR \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate $learning_rate \
    --projector_lr $projector_lr \
    --vision_lr $vision_lr \
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

# ... existing code ...
cp $OUT_DIR/model*  $MODEL_NAME/
python3 upload_model.py