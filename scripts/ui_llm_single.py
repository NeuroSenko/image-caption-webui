import gradio as gr

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


def process_single_image(
    llm_checkpoint,
    clip_checkpoint,
    image_adapter_checkpoint,
    prompt,
    image,
    max_new_tokens,
    stop_sequence,
    do_sample,
    top_k,
    top_p,
    temperature,
):
    clip_processor, clip_model = load_clip_model(clip_checkpoint)
    text_model, tokenizer = load_llm_model(llm_checkpoint)
    image_adapter = load_image_adapter(image_adapter_checkpoint, clip_model, text_model)

    image_features = get_image_features(image, clip_processor, clip_model)
    image_embedding = create_image_embedding_by_features(image_features, image_adapter)

    top_k = top_k if do_sample else None
    top_p = top_p if do_sample else None
    temperature = temperature if do_sample else None

    log("Processing image...")
    caption = process_prompt_with_image_embedding(
        prompt=prompt,
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

    log("Image processing is completed")
    return caption


def build_ui_llm_single():
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
                    info="Use [IMAGE] to specify image embedding position.",
                    value="[IMAGE] A descriptive caption for this image:\n",
                    interactive=True,
                )
                input_image = gr.Image(label="Image")
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
            with gr.Column():
                output_caption = gr.Textbox(label="Caption", interactive=False)
                submit_btn = gr.Button(
                    value="Submit",
                    variant="primary",
                )
                submit_btn.click(
                    fn=process_single_image,
                    inputs=[
                        input_llm,
                        input_clip,
                        input_image_adapter,
                        input_prompt,
                        input_image,
                        input_max_output_tokens,
                        input_stop_sequence,
                        input_do_sampling,
                        input_top_k,
                        input_top_p,
                        input_temperature,
                    ],
                    outputs=output_caption,
                )
