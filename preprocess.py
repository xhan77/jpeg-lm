# Example command: python preprocess.py --query_vllm_server "local" --prefix_ratio 0.375 --temp 1.0 --topp 0.9 --topk 50 --test_image_path 'example_image_input/*.png' --repeat_generation 10 --seed 42 --output_dir "out"

import torch # version 2.1.2
from torchvision import transforms # version 0.16.2
from PIL import Image
import io
import os
import glob
import argparse
import re
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, set_seed # version 4.38.2
import requests
import json
from packaging import version
import PIL # version 10.2.0
if version.parse(PIL.__version__) != version.parse("10.2.0"):
    raise ValueError("check Pillow version, should be 10.2.0 (at least 10.2.0, higher than 10.2.0 not tested)")


QUALITY = 25
TOTAL_PATCH_ROWS = 32 # how many [FFD0-FFD7] rows are there? for 256x256 resolution, 4:2:0 color -- (256 image side length / 16 patch side length / 8 restart markers) * (256 image side length / 16 patch side length) = 32
MAX_MODEL_SEQ_LEN = 12288
CACHE_TABLES_FN = "cache_tables.jpg"
UNICODE_OFFSET = 10240


def convert_img_to_bytes(img, quality=25, save_tables_file=False): 
    def ConvertToJpeg(im, quality):
        with io.BytesIO() as f: # get jpeg bytes without fixed file headers -- streamtype=2
            im.save(f, format='JPEG', quality=quality, subsampling="4:2:0", streamtype=2, restart_marker_blocks=1) 
            return f.getvalue()
    hexdata = ConvertToJpeg(img, quality).hex()
    int_list = [int(_e) for _e in bytearray.fromhex(hexdata)]
    str_list = ''.join([chr(_e + UNICODE_OFFSET) for _e in int_list]) # use braille characters to represent bytes -- UNICODE_OFFSET=10240
    if save_tables_file: # save jpeg headers (fixed tables for quantization and entropy coding) -- streamtype=1
        img.save(CACHE_TABLES_FN, quality=quality, subsampling="4:2:0", streamtype=1, restart_marker_blocks=1) 
    return str_list


def save_byte_image(byte_list, gen_img_fn):
    with open(CACHE_TABLES_FN, 'rb') as f:
        hexdata = f.read().hex()
        table_int_list = [int(_e) for _e in bytearray.fromhex(hexdata)]
        table_int_list = table_int_list[2:-2] # removing first 2 and last 2 bytes (FF D8 and FF D9)
    int_list = [ord(_e) - UNICODE_OFFSET for _e in byte_list]
    int_list = int_list[:2] + table_int_list + int_list[2:]
    new_hexdata = bytes(int_list)
    with open(gen_img_fn + '.jpg', 'wb') as f:
        f.write(new_hexdata)


def build_prompt(image_str, tokenizer, prefix_ratio=0.0):
    def trim_after_n_occurrences(input_string, substring, n):
        pattern = re.compile('(' + re.escape(substring) + '(?:.*?' + re.escape(substring) + '){' + str(n-1) + '})')
        match = pattern.search(input_string)
        return input_string[:match.end()] if match else input_string

    if prefix_ratio == 0.0:
        input_ids = [tokenizer.bos_token_id]
    elif prefix_ratio > 0:
        image_str = trim_after_n_occurrences(image_str, '\u28ff\u28d7', int(TOTAL_PATCH_ROWS * prefix_ratio)) # UNICODE_OFFSET = 10240
        input_ids = [tokenizer.bos_token_id] + [_e  for _e in tokenizer(image_str, add_special_tokens=False)['input_ids']]
    else:
        raise ValueError(f"invalid prefix ratio")
    return input_ids


def clean_image_string(output_image_str): # UNICODE_OFFSET = 10240
    matched_output_image_str = re.match('\u28ff\u28d8.*?\u28ff\u28d9', output_image_str)
    if matched_output_image_str:
        output_image_str = matched_output_image_str.group()
    else:
        if output_image_str[-2:] != '\u28ff\u28d9':
            output_image_str = output_image_str + '\u28ff\u28d9'
        if output_image_str[:2] != '\u28ff\u28d8':
            output_image_str = '\u28ff\u28d8' + output_image_str
    return output_image_str


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model_name_or_path", type=str, default="uwnlp/jpeg-lm-reimplementation", help="")
    parser.add_argument("--prefix_ratio", type=float, default=0.5, help="")
    parser.add_argument("--temp", type=float, default=1.0, help="")
    parser.add_argument("--topp", type=float, default=0.9, help="")
    parser.add_argument("--topk", type=int, default=50, help="")
    parser.add_argument("--query_vllm_server", type=str, default=None, help="")
    parser.add_argument("--test_image_path", type=str, default=None, help="")
    parser.add_argument("--output_dir", type=str, default=None, help="")
    parser.add_argument("--repeat_generation", type=int, default=1, help="")
    parser.add_argument("--seed", type=int, default=2024, help="")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(args.seed)

    preprocess = transforms.Compose([
        transforms.RandomResizedCrop((256, 256), scale=(1.0, 1.0), ratio=(1.0, 1.0), antialias=True),
    ]) # work with 256x256 resolution for now

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False, add_eos_token=True)

    if args.query_vllm_server:
        from vllm import LLM, SamplingParams # version 0.3.3
        if args.query_vllm_server == "local":
            model = LLM(model=args.model_name_or_path, max_context_len_to_capture=MAX_MODEL_SEQ_LEN, dtype='float16')
    else:
        jpeg_model_name = args.model_name_or_path
        config = AutoConfig.from_pretrained(jpeg_model_name)
        config.use_cache = True # use_cache=False was used in training
        model = AutoModelForCausalLM.from_pretrained(
            jpeg_model_name,
            config=config,
            ignore_mismatched_sizes=False,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map='auto',
        )

    prefix_ratio = args.prefix_ratio
    temp, topp, topk = args.temp, args.topp, args.topk
    config_str = f"ratio{prefix_ratio}_temp{temp}_topp{topp}_topk{topk}"

    image_path = sorted(glob.glob(args.test_image_path))
    reader = []
    for _image_path in image_path:
        image = Image.open(_image_path).convert('RGB')
        reader.append({'image': preprocess(image)})
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        args.output_dir = "./"

    for _i, batch in enumerate(reader):
        print(f"instance {_i} ...")

        image_str = convert_img_to_bytes(batch['image'], quality=QUALITY, save_tables_file=True) # can set save_tables_file=False if the tables were saved before (under same config), may be useful in multiprocessing
        partial_input_ids = build_prompt(image_str, tokenizer, prefix_ratio=prefix_ratio)
        genlen = max(MAX_MODEL_SEQ_LEN - len(partial_input_ids), 0)
        partial_input_ids_tensor = torch.tensor(partial_input_ids, dtype=torch.long).to(device).repeat(args.repeat_generation, 1)
        partial_input_string = tokenizer.decode(partial_input_ids, skip_special_tokens=True).strip()

        if genlen <= 0:
            print("skipping due to genlen <= 0")
            continue
        with torch.no_grad():
            if args.query_vllm_server is None:
                outputs = model.generate(inputs=partial_input_ids_tensor, max_new_tokens=genlen, temperature=temp, top_p=topp, top_k=topk, do_sample=True, min_new_tokens=1)
                output_string_list = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            elif args.query_vllm_server == "local":
                vllm_rt_data = model.generate(prompt_token_ids=partial_input_ids_tensor.tolist(), sampling_params=SamplingParams(max_tokens=genlen, temperature=temp, top_p=topp, top_k=topk, spaces_between_special_tokens=False))
                output_string_list = []
                for choice in vllm_rt_data:
                    output_string_list.append(partial_input_string + choice.outputs[0].text.strip())
            else:
                response = requests.post(f"http://{args.query_vllm_server}/v1/completions", headers={}, json={'model': "jpeglm", 'prompt': partial_input_ids_tensor.tolist(), 'max_tokens': genlen, 'temperature': temp, 'top_p': topp, 'top_k': topk, 'spaces_between_special_tokens': False, 'stream': False,}, stream=False)
                vllm_rt_data = json.loads(response.content)
                output_string_list = []
                for choice in vllm_rt_data['choices']:
                    output_string_list.append(partial_input_string + choice['text'].strip())

        # original image (after tokenization and de-tokenization)
        save_byte_image(image_str, os.path.join(args.output_dir, f"original_{_i}_{config_str}"))

        # partial image (prompt)
        save_byte_image(clean_image_string(partial_input_string), os.path.join(args.output_dir, f"partial_{_i}_{config_str}"))

        # generation
        for _j, output_string in enumerate(output_string_list):
            save_byte_image(clean_image_string(output_string), os.path.join(args.output_dir, f"generation_{_i}-{_j}_{config_str}"))


if __name__ == "__main__":
    main()