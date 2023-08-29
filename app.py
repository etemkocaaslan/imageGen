import torch
from PIL import Image
import gradio as gr
from diffusers import StableDiffusionImg2ImgPipeline, DiffusionPipeline
from gradio import (Blocks, Row, Button, Image, ClearButton, UploadButton,
                    Markdown, Column, Textbox, Slider, Tabs, TabItem, Dropdown)
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message="`text_config_dict` is provided which will be used to initialize `CLIPTextConfig`.*")


model = "runwayml/stable-diffusion-v1-5"


def generate_img_2_img(image, prompt, strength=0.75, guidance_scale=7.5):
    print(prompt)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    image = image.resize((768, 512))
    images = pipe(prompt=prompt, image=image, strength=strength, guidance_scale=guidance_scale).images
    return images[0]


def generate_txt_2_img(prompt):
    print(prompt)
    pipe = DiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    images = pipe(prompt=prompt).images
    return images[0]


def create_ui():
    with Blocks() as demo:
        with Tabs():
            create_webcam_tab()
            create_upload_tab()
            create_text_tab()
    
    return demo


def create_webcam_tab():
    with TabItem(label='Webcam'):
        with Row():
            with Column():
                webcam_img = Image(source='webcam', type='pil')
                text_input = Textbox()
                slider_strength = Slider(0.05, 1, value=0.75, step=0.05, label='Strength',
                                        info="Indicates extent to transform it")
                slider_guidance = Slider(1, 10, value=7.5, step=0.5, label='Guidance_scale',
                                         info="Higher guidance scale, Enable when > 1 ")
            with Column():
                output_img = Image(type='pil')
                generate_btn = Button(value="Generate")
                generate_btn.click(generate_img_2_img, inputs=[webcam_img, text_input, slider_strength, slider_guidance],
                                   outputs=[output_img])
                clear_btn = ClearButton([output_img])


def create_upload_tab():
    with TabItem(label='Upload'):
        with Row():
            with Column():
                upload_img = Image(source='upload', type='pil')
                text_input = Textbox()
                slider_strength = Slider(0.05, 1, value=0.75, step=0.05, label='Strength',
                                        info="Indicates extent to transform it")
                slider_guidance = Slider(1, 10, value=7.5, step=0.5, label='Guidance_scale',
                                         info="Higher guidance scale, Enable when > 1 ")
            with Column():
                output_img = Image(type='pil')
                generate_btn = Button(value="Generate")
                generate_btn.click(generate_img_2_img, inputs=[upload_img, text_input, slider_strength, slider_guidance],
                                   outputs=[output_img])
                clear_btn = ClearButton([output_img])


def create_text_tab():
    with TabItem(label='Text'):
        with Row():
            with Column():
                text_prompt = Textbox()
            with Column():
                output_img = Image(type='pil')
                generate_btn = Button(value="Generate")
                generate_btn.click(generate_txt_2_img, inputs=text_prompt, outputs=[output_img])
                clear_btn = ClearButton([output_img])


if __name__ == "__main__":
    demo_app = create_ui()
    demo_app.close()
    demo_app.launch(share=True)

