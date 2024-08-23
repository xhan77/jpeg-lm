# JPEG-LM Preprocessing

JPEG-LM autoregressively generates image file bytes like language. Data preprocessing in JPEG-LM is very simple and can be done with `preprocess.py`. 

Example command: `python preprocess.py --query_vllm_server "local" --prefix_ratio 0.375 --temp 1.0 --topp 0.9 --topk 50 --test_image_path 'example_image_input/*.png' --repeat_generation 10 --seed 42 --output_dir "out"`. 