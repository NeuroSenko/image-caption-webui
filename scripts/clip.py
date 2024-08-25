import gc
import os
import torch
from transformers import AutoModel, AutoProcessor
from scripts.logger import log
from PIL import Image

loaded_checkpoint_name = None
clip_processor = None
clip_model = None


def load_clip_model(checkpoint_name: str):
    global loaded_checkpoint_name
    global clip_processor
    global clip_model

    if loaded_checkpoint_name == checkpoint_name:
        return [clip_processor, clip_model]

    unload_clip_model()

    log(f"[CLIP] Loading checkpoint {checkpoint_name}")
    checkpoint_path = os.path.join("models", "CLIP", checkpoint_name)

    clip_processor = AutoProcessor.from_pretrained(checkpoint_path)
    clip_model = AutoModel.from_pretrained(checkpoint_path)
    clip_model = clip_model.vision_model
    clip_model.eval()
    clip_model.requires_grad_(False)
    clip_model.to("cuda")

    loaded_checkpoint_name = checkpoint_name

    log(f"[CLIP] Checkpoint {checkpoint_path} is loaded")
    return [clip_processor, clip_model]


def unload_clip_model():
    global loaded_checkpoint_name
    global clip_processor
    global clip_model

    if "loaded_checkpoint_name" not in globals() or loaded_checkpoint_name is None:
        return

    log(f"[CLIP] Unloading checkpoint {loaded_checkpoint_name}")

    loaded_checkpoint_name = None
    clip_processor = None
    clip_model = None

    torch.cuda.empty_cache()
    gc.collect()

    log(f"[CLIP] Checkpoint is unloaded")


def get_image_features(input_image: Image.Image, clip_processor, clip_model):
    image = clip_processor(images=input_image, return_tensors="pt").pixel_values
    image = image.to("cuda")
    with torch.amp.autocast_mode.autocast("cuda", enabled=True):
        vision_outputs = clip_model(pixel_values=image, output_hidden_states=True)
        image_features = vision_outputs.hidden_states[-2]
        return image_features


def load_clip_models_list():
    checkpoints_path = os.path.join("models", "CLIP")
    return next(os.walk(checkpoints_path))[1]
