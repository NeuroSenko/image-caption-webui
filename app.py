import gradio as gr
import torch
from scripts.logger import log
from scripts.ui_llm_batch import build_ui_llm_batch
from scripts.ui_llm_single import build_ui_llm_single

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

log(f"Image caption WebUI start")
log(f"Use device: {torch.cuda.get_device_name(0)}")

with gr.Blocks(title="Image Caption WebUI") as demo:
    with gr.Tab("LLM Single"):
        build_ui_llm_single()
    with gr.Tab("LLM Batch"):
        build_ui_llm_batch()

demo.launch()
