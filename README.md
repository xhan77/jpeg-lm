# JPEG-LM Preprocessing

JPEG-LM autoregressively generates image file bytes like language. Data preprocessing in JPEG-LM is very simple and can be done with `preprocess.py`. 

Example command: `python preprocess.py --query_vllm_server "local" --prefix_ratio 0.375 --temp 1.0 --topp 0.9 --topk 50 --test_image_path 'example_image_input/*.png' --repeat_generation 10 --seed 42 --output_dir "out"`. 

Note that we use `pillow==10.2.0` (lower versions won't work). `torch (2.1.2)`, `transformers (4.38.2)`, and `vllm (0.3.3)` should be installed as well. Different sampling hyperparameters can also be tried further (e.g., removing top-p for landscape images).