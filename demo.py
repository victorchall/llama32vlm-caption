# Quick implementation for testing generate parameters
# based on snippet provided by huggingface: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision

import requests
import time
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor


path = "/mnt/stuff/1234.webp"
image = Image.open(path)

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)
processor = AutoProcessor.from_pretrained(model_id)

prompt = "Would this image be appropriate as a desktop wallpaper in a work environment in the United States?"

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": f"{prompt}"}
    ]}
]
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

for i in range(0,1):
    start_perf = time.perf_counter()
    inputs = processor(image, input_text, return_tensors="pt").to(model.device)

    # adjust parameters for testing...
    output = model.generate(**inputs, max_new_tokens=350, do_sample=True, temperature=0.9, num_beams=2)

    print(processor.decode(output[0]))
    stop_perf = time.perf_counter()
    print(f"      ** elapsed: {stop_perf - start_perf:.2f}")