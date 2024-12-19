from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import json
import os
import argparse

parser = argparse.ArgumentParser(description="test")

parser.add_argument('--model_path', default="Molmo-7B-D-0924", type=str, help='model path')
parser.add_argument('--json_file_path', default="data/chunk01_enhanced.json", type=str, help='eval data path')
parser.add_argument('--img_folder_path', default="data/images", type=str, help='img path')

args = parser.parse_args()


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

with open(args.json_file_path, 'r') as f:
    all_data = json.load(f)

for single_data in all_data:
    img_path = os.path.join(args.img_folder_path, single_data['image'])
    question = single_data['conversations'][0]['value']
    gold_answer = single_data['conversations'][1]['value']
    
    inputs = processor.process(
        images=[Image.open(img_path)],
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

    print(question, generated_answer, gold_answer)
    
    x = input()