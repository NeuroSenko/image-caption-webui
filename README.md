# Image Caption WebUI

![](https://files.catbox.moe/pmh8dw.jpg)

## Installation
1. `git clone https://github.com/NeuroSenko/image-caption-webui.git`
2. Run `install.bat` in order to init venv and install python deps
3. Download LLM model and put it to `/models/LLM` folder:  
You can use [Meta-LLama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B/tree/main) ([not-gated mirror](https://huggingface.co/unsloth/Meta-Llama-3.1-8B/tree/main)) or any of it's finetune like [Hermes-3-Llama-3.1-8B](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B/tree/main)
4. Download [CLIP](https://huggingface.co/google/siglip-so400m-patch14-384/tree/main) model and put it to `/models/CLIP` folder
5. Download [Image Adapter](https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha/tree/main/wpkklhc6) and put it to `/models/image_adapter` folder

Models folder structure should be something like this:
```
.
├── ...
├── models
│   └── CLIP
│   │   └── siglip-so400m-patch14-384
│   │       ├── config.json
│   │       ├── model.safetensors
│   │       └── ...
│   ├── image_adapter
│   │   └── joy-caption-1
│   │       ├── config.yaml
│   │       └── image_adapter.pt
│   └── LLM
│       └── Meta-Llama-3.1-8B
│           ├── config.json
│           ├── model-00001-of-00004.safetensors
│           ├── tokenizer.json
│           └── ...
└── ...
```
6. Run `start.bat` to run the program

## Troubleshooting

If you get `Torch not compiled with CUDA enabled` error, try these two commands:
```
.\venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu118
```
