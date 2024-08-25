import gc
import os
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoModelForCausalLM,
)
import torch
import torch.amp.autocast_mode

from scripts.logger import log

loaded_checkpoint_name = None
text_model = None
tokenizer = None


def load_llm_model(checkpoint_name: str):
    global loaded_checkpoint_name
    global text_model
    global tokenizer

    if loaded_checkpoint_name == checkpoint_name:
        return [text_model, tokenizer]

    unload_llm_model()

    checkpoint_path = os.path.join("models", "LLM", checkpoint_name)

    log(f"[LLM] Loading checkpoint {checkpoint_name}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=False)
    assert isinstance(tokenizer, PreTrainedTokenizer) or isinstance(
        tokenizer, PreTrainedTokenizerFast
    ), f"Tokenizer is of type {type(tokenizer)}"

    text_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, device_map="auto", torch_dtype=torch.bfloat16
    )
    text_model.eval()

    loaded_checkpoint_name = checkpoint_name

    log(f"[LLM] Checkpoint {checkpoint_name} is loaded")
    return [text_model, tokenizer]


def unload_llm_model():
    global loaded_checkpoint_name
    global text_model
    global tokenizer

    if "loaded_checkpoint_name" not in globals() or loaded_checkpoint_name is None:
        return

    log(f"[LLM] Unloading checkpoint {loaded_checkpoint_name}")

    loaded_checkpoint_name = None
    text_model = None
    tokenizer = None

    torch.cuda.empty_cache()
    gc.collect()

    log(f"[LLM] Checkpoint is unloaded")


def process_prompt_with_image_embedding(
    prompt: str,
    image_embedding,
    text_model,
    tokenizer,
    max_new_tokens=300,
    stop_sequence=None,
    do_sample=True,
    top_k=10,
    top_p=0.9,
    temperature=0.5,
):
    if "[IMAGE]" in prompt:
        [prompt_before, prompt_after] = prompt.split("[IMAGE]")

    if prompt_before is None or len(prompt_before) == 0:
        prompt_before = " "

    if prompt_after is None or len(prompt_after) == 0:
        prompt_after = " "

    # Tokenize the prompt
    prompt_before_tokenized = tokenizer.encode(
        prompt_before,
        return_tensors="pt",
        padding=False,
        truncation=False,
        add_special_tokens=False,
    )

    prompt_after_tokenized = tokenizer.encode(
        prompt_after,
        return_tensors="pt",
        padding=False,
        truncation=False,
        add_special_tokens=False,
    )

    # Embed prompt
    prompt_before_embeds = text_model.model.embed_tokens(
        prompt_before_tokenized.to("cuda")
    )
    prompt_after_embeds = text_model.model.embed_tokens(
        prompt_after_tokenized.to("cuda")
    )

    embedded_bos = text_model.model.embed_tokens(
        torch.tensor(
            [[tokenizer.bos_token_id]], device=text_model.device, dtype=torch.int64
        )
    )

    # Construct prompts
    inputs_embeds = torch.cat(
        [
            embedded_bos.expand(image_embedding.shape[0], -1, -1),
            prompt_before_embeds.expand(image_embedding.shape[0], -1, -1),
            image_embedding.to(dtype=embedded_bos.dtype),
            prompt_after_embeds.expand(image_embedding.shape[0], -1, -1),
        ],
        dim=1,
    )

    input_ids = torch.cat(
        [
            torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long),
            prompt_before_tokenized,
            torch.zeros((1, image_embedding.shape[1]), dtype=torch.long),
            prompt_after_tokenized,
        ],
        dim=1,
    ).to("cuda")
    attention_mask = torch.ones_like(input_ids)

    generate_ids = text_model.generate(
        input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        suppress_tokens=None,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Trim off the prompt
    generate_ids = generate_ids[:, input_ids.shape[1] :]
    if generate_ids[0][-1] == tokenizer.eos_token_id:
        generate_ids = generate_ids[:, :-1]

    caption = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )[0]

    if stop_sequence is not None and len(stop_sequence) > 0:
        caption = caption.split(stop_sequence)[0]

    return caption.strip()


def load_llm_models_list():
    checkpoints_path = os.path.join("models", "LLM")
    return next(os.walk(checkpoints_path))[1]
