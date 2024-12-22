
model_versions=("Molmo7b_CN_v11" "Molmo7b_CN_v9" "Molmo7b_CN_v8" "Molmo7b_lora_CN_v8" "Molmo7b_CN_v0" "Molmo7b_CN_v6_epoch3")

# 下载模型
for model_version in "${model_versions[@]}"; do
    python3 down_model.py --model_version $model_version
done

# 开始测试
i=0
for model_version in "${model_versions[@]}"; do
    CUDA_VISIBLE_DEVICES=$i python3 evaluation/eval_main.py \
        --model_path "${model_version}" \
        --data_path data/ground_test/evaluation_01.parquet \
        --img_folder_path "data/images" \
        --out_path "output/Molmo7b_CN_v${model_version}_ground.txt" &
    
    i=$((i + 1))
done

wait # 等待所有后台进程完成