import glob
import os
from pathlib import Path
import gradio as gr
from PIL import Image
from tqdm import tqdm

from scripts.clip import (
    get_image_features,
    load_clip_model,
    load_clip_models_list,
    unload_clip_model,
)
from scripts.image_adapter import (
    create_image_embedding_by_features,
    load_image_adapter,
    load_image_adapter_models_list,
    unload_image_adapter,
)
from scripts.llm import (
    load_llm_model,
    load_llm_models_list,
    process_prompt_with_image_embedding,
    unload_llm_model,
)
from scripts.logger import log


def load_checkpoints(llm_checkpoint, clip_checkpoint, image_adapter_checkpoint):
    _, clip_model = load_clip_model(clip_checkpoint)
    text_model, _ = load_llm_model(llm_checkpoint)
    load_image_adapter(image_adapter_checkpoint, clip_model, text_model)


def unload_checkpoints():
    log("Unloading checkpoints")

    unload_llm_model()
    unload_clip_model()
    unload_image_adapter()

    log("All checkpoints are unloaded")


def get_caption_path(template: str, image_path: str):
    template = template.replace("[IMAGE_DIR]", os.path.dirname(image_path))
    template = template.replace("[IMAGE_FULL_NAME]", image_path)
    template = template.replace(
        "[IMAGE_FULL_NAME_WITHOUT_EXTENSION]", os.path.splitext(image_path)[0]
    )
    template = template.replace("[IMAGE_NAME]", os.path.basename(image_path))
    template = template.replace("[IMAGE_NAME_WITHOUT_EXTENSION]", Path(image_path).stem)
    return template


def process_multiple_images(
    llm_checkpoint,
    clip_checkpoint,
    image_adapter_checkpoint,
    prompt,
    images_path,
    caption_save_path,
    search_in_nested_folders,
    log_details_in_terminal,
    apply_additional_caption,
    additional_caption_path,
    max_new_tokens,
    stop_sequence,
    do_sample,
    top_k,
    top_p,
    temperature,
    progress=gr.Progress(),
):
    if images_path is None or len(images_path) == 0:
        log("Images path is not specified")
        return "Images path is not specified"

    clip_processor, clip_model = load_clip_model(clip_checkpoint)
    text_model, tokenizer = load_llm_model(llm_checkpoint)
    image_adapter = load_image_adapter(image_adapter_checkpoint, clip_model, text_model)

    top_k = top_k if do_sample else None
    top_p = top_p if do_sample else None
    temperature = temperature if do_sample else None

    images_list = []

    search_path = (
        f"{images_path}/**/*" if search_in_nested_folders else f"{images_path}/*"
    )

    for file_path in glob.iglob(search_path, recursive=True):
        if (
            file_path.endswith(".jpg")
            or file_path.endswith(".jpeg")
            or file_path.endswith(".png")
        ):
            images_list.append(file_path)

    log(f"Total {len(images_list)} images in queue...")

    for image_path in tqdm(
        progress.tqdm(images_list, desc="Images processing"), desc="Images processing"
    ):
        image = Image.open(image_path)
        image_features = get_image_features(image, clip_processor, clip_model)
        image_embedding = create_image_embedding_by_features(
            image_features, image_adapter
        )

        image_prompt = prompt

        if apply_additional_caption:
            caption_path = get_caption_path(additional_caption_path, image_path)

            with open(caption_path, encoding="utf-8") as f:
                additional_caption = f.read()
                image_prompt = image_prompt.replace("[CAPTION]", additional_caption)

        if log_details_in_terminal:
            log(f"Prompt: {image_prompt}")

        caption = process_prompt_with_image_embedding(
            prompt=image_prompt,
            image_embedding=image_embedding,
            text_model=text_model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            stop_sequence=stop_sequence,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )

        if log_details_in_terminal:
            log(f"Caption: {caption}")

        image_caption_save_path = get_caption_path(caption_save_path, image_path)

        with open(image_caption_save_path, "w", encoding="utf-8") as f:
            f.write(caption)

    log("Images processing is completed")
    return "Images processing is completed"


def build_ui_llm_batch():
    llm_models_list = load_llm_models_list()
    default_llm_model = llm_models_list[0] if len(llm_models_list) > 0 else None

    clip_models_list = load_clip_models_list()
    default_clip_model = clip_models_list[0] if len(clip_models_list) > 0 else None

    image_adapter_models_list = load_image_adapter_models_list()
    default_image_adapter_model = (
        image_adapter_models_list[0] if len(image_adapter_models_list) > 0 else None
    )

    with gr.Blocks():
        with gr.Row():
            input_llm = gr.Dropdown(
                label="LLM", choices=llm_models_list, value=default_llm_model
            )
            input_clip = gr.Dropdown(
                label="CLIP", choices=clip_models_list, value=default_clip_model
            )
            input_image_adapter = gr.Dropdown(
                label="Image Adapter",
                choices=image_adapter_models_list,
                value=default_image_adapter_model,
            )
            with gr.Column():
                load_checkpoints_btn = gr.Button("Load checkpoints")
                load_checkpoints_btn.click(
                    fn=load_checkpoints,
                    inputs=[
                        input_llm,
                        input_clip,
                        input_image_adapter,
                    ],
                )
                unload_checkpoints_btn = gr.Button("Unload checkpoints")
                unload_checkpoints_btn.click(fn=unload_checkpoints)
        with gr.Row():
            with gr.Column():
                input_prompt = gr.Textbox(
                    label="Prompt",
                    info="Use [IMAGE] to specify image embedding position. Use [CAPTION] to specify position of additional caption.",
                    value="[IMAGE] A descriptive caption for this image:\n",
                    interactive=True,
                )
                input_images_path = gr.Textbox(
                    label="Images path",
                    placeholder="C:\\datasets\\waifu",
                    interactive=True,
                )
                input_caption_save_path = gr.Textbox(
                    label="Captions save path",
                    value="[IMAGE_FULL_NAME_WITHOUT_EXTENSION].nlp.txt",
                    interactive=True,
                    info="See path editor note below",
                )
                input_search_in_nested_folders = gr.Checkbox(
                    label="Search images in nested folders", value=True
                )
                input_log_details_in_terminal = gr.Checkbox(
                    label="Log details in terminal", value=False
                )
                with gr.Accordion(label="Additional caption", open=False):
                    input_apply_additional_caption = gr.Checkbox(
                        label="Enable", value=False
                    )
                    input_additional_caption_path = gr.Textbox(
                        label="Additional caption path",
                        value="[IMAGE_FULL_NAME_WITHOUT_EXTENSION].tags.txt",
                        info="See path editor note below",
                    )
                    gr.HTML(
                        """Additional captions are used when you already have some captions and want to use<br>
                        them to generate another caption.<br>
                        <br>
                        For example, it will be useful if you want to use tags extracted by WD-Tagger in your prompt.
                    """
                    )
                with gr.Accordion(label="LLM Params", open=False):
                    input_max_output_tokens = gr.Slider(
                        label="Max output tokens",
                        minimum=0,
                        maximum=1000,
                        value=300,
                    )
                    input_stop_sequence = gr.Textbox(
                        label="Stop Sequence",
                        placeholder="[EOS]",
                        interactive=True,
                    )
                    input_do_sampling = gr.Checkbox(label="Do sampling", value=True)
                    input_top_k = gr.Slider(
                        label="Top K", minimum=0, maximum=100, value=10
                    )
                    input_top_p = gr.Slider(
                        label="Top P", minimum=0, maximum=1, value=0.9
                    )
                    input_temperature = gr.Slider(
                        label="Temperature", minimum=0, maximum=1, value=0.5
                    )

                    def do_sampling_change(do_sampling):
                        return [
                            gr.Slider(interactive=do_sampling),
                            gr.Slider(interactive=do_sampling),
                            gr.Slider(interactive=do_sampling),
                        ]

                    input_do_sampling.change(
                        fn=do_sampling_change,
                        inputs=[input_do_sampling],
                        outputs=[input_top_k, input_top_p, input_temperature],
                    )

                with gr.Accordion(label="Path editor note", open=False):
                    gr.HTML(
                        """When handling an image with name "C:\\datasets\\waifu\\123456.png"<br>
                            next substrings will be transformed in path template:<br>
                            <br>
                            <ul>
                                <li>[IMAGE_DIR] ➡ C:\\datasets\\waifu</li>
                                <li>[IMAGE_FULL_NAME] ➡ C:\\datasets\\waifu\\123456.png</li>
                                <li>[IMAGE_FULL_NAME_WITHOUT_EXTENSION] ➡ C:\\datasets\\waifu\\123456</li>
                                <li>[IMAGE_NAME] ➡ 123456.png</li>
                                <li>[IMAGE_NAME_WITHOUT_EXTENSION] ➡ 123456</li>
                            </ul>
                    """
                    )
            with gr.Column():
                output_result = gr.Textbox(label="Result")
                submit_btn = gr.Button(
                    value="Submit",
                    variant="primary",
                )
                submit_btn.click(
                    process_multiple_images,
                    [
                        input_llm,
                        input_clip,
                        input_image_adapter,
                        input_prompt,
                        input_images_path,
                        input_caption_save_path,
                        input_search_in_nested_folders,
                        input_log_details_in_terminal,
                        input_apply_additional_caption,
                        input_additional_caption_path,
                        input_max_output_tokens,
                        input_stop_sequence,
                        input_do_sampling,
                        input_top_k,
                        input_top_p,
                        input_temperature,
                    ],
                    output_result,
                )
