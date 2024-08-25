import gc
from torch import nn
import torch
import torch.amp.autocast_mode
import os

from scripts.logger import log


class ImageAdapter(nn.Module):
    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)

    def forward(self, vision_outputs: torch.Tensor):
        x = self.linear1(vision_outputs)
        x = self.activation(x)
        x = self.linear2(x)
        return x


loaded_checkpoint_name = None
image_adapter = None


def load_image_adapter(checkpoint_name: str, clip_model, text_model):
    global loaded_checkpoint_name
    global image_adapter

    if loaded_checkpoint_name == checkpoint_name:
        return image_adapter

    unload_image_adapter()

    log(f"[Image Adapter] Loading checkpoint {checkpoint_name}")

    checkpoint_path = os.path.join(
        "models", "image_adapter", checkpoint_name, "image_adapter.pt"
    )

    image_adapter = ImageAdapter(
        clip_model.config.hidden_size, text_model.config.hidden_size
    )

    image_adapter.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )

    image_adapter.eval()
    image_adapter.to("cuda")

    loaded_checkpoint_name = checkpoint_name

    log(f"[Image Adapter] Checkpoint {checkpoint_name} is loaded")
    return image_adapter


def unload_image_adapter():
    global loaded_checkpoint_name
    global image_adapter

    if "loaded_checkpoint_name" not in globals() or loaded_checkpoint_name is None:
        return

    log(f"[Image Adapter] Unloading checkpoint {loaded_checkpoint_name}")

    loaded_checkpoint_name = None
    image_adapter = None

    torch.cuda.empty_cache()
    gc.collect()

    log(f"[Image Adapter] Checkpoint is unloaded")


def create_image_embedding_by_features(image_features, image_adapter):
    with torch.amp.autocast_mode.autocast("cuda", enabled=True):
        embedded_images = image_adapter(image_features)
        embedded_images = embedded_images.to("cuda")
        return embedded_images


def load_image_adapter_models_list():
    checkpoints_path = os.path.join("models", "image_adapter")
    return next(os.walk(checkpoints_path))[1]
