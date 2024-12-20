from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import json
import os
import argparse
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description="test")
parser.add_argument('--model_path', default="Molmo-7B-D-0924", type=str, help='model path')
parser.add_argument('--parquet_file_path', default="data/chunk01.parquet", type=str, help='eval data path')
parser.add_argument('--img_folder_path', default="data/images", type=str, help='img path')
args = parser.parse_args()

def scale_bbox(bbox, img_size):
    width, height = img_size
    x1, y1, x2, y2 = bbox
    x1 = x1/width * 100
    y1 = y1/height * 100
    x2 = x2/width * 100
    y2 = y2/height * 100
    return x1, y1, x2, y2

def is_in_rectangle(box, point):
    x, y = point
    x1, y1, x2, y2 = box
    return (x1 <= x <= x2 and y1 <= y <= y2)

def parse_point(point_str):
    """
    point_str will be ' <point x="49.8" y="69.8"></point> '
    """
    x_start = point_str.find('x="') + 3
    x_end = point_str.find('"', x_start)
    y_start = point_str.find('y="') + 3
    y_end = point_str.find('"', y_start)
    x = float(point_str[x_start:x_end])
    y = float(point_str[y_start:y_end])
    return x, y

processor = AutoProcessor.from_pretrained(
    args.model_path,
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

all_data = pd.read_parquet(args.parquet_file_path)

success_count = 0
total_count = 0
error_count = 0
for idx, single_data in tqdm(all_data.iterrows()):
    img_path = os.path.join(args.img_folder_path, single_data['view']+'.png')
    question = f"""You are going to refer an element on the UI interface of the image, following the instruction.\nPlease point out the element you want to refer to complete the instruction.\n\nYour instruction:\nwhere is 位于{single_data['abs_pos']}，并且{single_data['rel_pos']}的{single_data['description']}?\nNote:\n1. Remember always to give a point to refer an element!\n2. Your response should be like this: ' <point x="?" y="?" alt="?">?</point>\ '\nBegin!"""
    
    img = Image.open(img_path)
    inputs = processor.process(
        images=[img],
        text=question
    )

    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer
    )


    generated_tokens = output[0,inputs['input_ids'].size(1):]
    generated_answer = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    try:
        point = parse_point(generated_answer)

        bbox = single_data['bbox_x0'], single_data['bbox_y0'], single_data['bbox_x1'], single_data['bbox_y1']
        bbox = scale_bbox(bbox, img.size)

        if is_in_rectangle(bbox, point):
            success_count += 1
    except:
        error_count += 1
    total_count += 1
print(f"ACC: {success_count/total_count:.2f}, error_count: {error_count}")
