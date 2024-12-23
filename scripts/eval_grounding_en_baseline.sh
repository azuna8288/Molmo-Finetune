

CUDA_VISIBLE_DEVICES=7 python3 evaluation/eval_main.py \
    --model_path Molmo-7B-D-0924 \
    --data_path data/grounding/ground_test/chunk03.parquet \
    --img_folder_path "data/images" \
    --out_path "output/Molmo-7B-D-0924_ground.txt"