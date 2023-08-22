import torch
from PIL import Image
import gradio as gr
from diffusers import StableDiffusionImg2ImgPipeline
from gradio import Blocks, Row, Button, Image, ClearButton, UploadButton, Markdown, Column, Textbox, Slider, Tabs, TabItem, Dropdown

models = ""
def generate(image, prompt, strength = 0.75, guidance_scale =7.5):
    print(prompt)  
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    image = image.resize((768, 512))
    images = pipe(prompt=prompt, image=image, strength=strength, guidance_scale=guidance_scale).images
    return images[0]


with Blocks() as demo:
    with Tabs():
        with TabItem(label = 'Webcam'):
            with Row():
                with Column():
                    webcamIm = Image(source = 'webcam', type='pil')
                    text_input = Textbox()
                    slider_input_s = Slider(0.75, 1, value= 0.05, step = 0.05, label = 'Strength', info = "Indicates extent to transform it")
                    slider_input_g = Slider(1, 10, value= 7.5, step = 0.5, label = 'Guidance_scale', info = "Higher guidance scale, Enable when > 1 ")
                with Column():
                    output_im = Image(type='pil')
                    btn_cap = Button(value="Generate")
                    btn_cap.click(generate, inputs = [webcamIm, text_input, slider_input_s, slider_input_g], outputs=[output_im])
                    btn_cl = ClearButton([output_im])
        with TabItem(label = 'Upload'):
            with Row():
                    with Column():
                        up_im = Image(source = 'upload', type='pil')
                        text_input = Textbox()
                        slider_input_s = Slider(0.75, 1, value= 0.05, step = 0.05, label = 'Strength', info = "Indicates extent to transform it")
                        slider_input_g = Slider(1, 10, value= 7.5, step = 0.5, label = 'Guidance_scale', info = "Higher guidance scale, Enable when > 1 ")
                    with Column():
                        output_im = Image(type='pil')
                        btn_cap = Button(value="Generate")
                        btn_cap.click(generate, inputs = [up_im, text_input, slider_input_s, slider_input_g], outputs=[output_im])
                        btn_cl = ClearButton([output_im])
        with TabItem(label = 'Text'):
            with Row():
                    with Column():
                        text_input = Textbox()
                        slider_input_s = Slider(0.75, 1, value= 0.05, step = 0.05, label = 'Strength', info = "Indicates extent to transform it")
                        slider_input_g = Slider(1, 10, value= 7.5, step = 0.5, label = 'Guidance_scale', info = "Higher guidance scale, Enable when > 1 ")
                        dropdwn_model = Dropdown()
                    with Column():
                        output_im = Image(type='pil')
                        btn_cap = Button(value="Generate")
                        btn_cap.click(generate, inputs = [up_im, text_input, slider_input_s, slider_input_g], outputs=[output_im])
                        btn_cl = ClearButton([output_im])
if __name__ == "__main__":
    
    demo.launch(share = True)