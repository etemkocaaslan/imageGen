Stable Diffusion Image Generation with Gradio

This repository contains a Python script that utilizes the Stable Diffusion model for image generation, along with a Gradio interface for easy interaction. The Stable Diffusion model is powered by the "StableDiffusionImg2ImgPipeline" from RunwayML, enabling users to transform images based on prompts and parameters.
Table of Contents

    Overview
    Dependencies
    Usage
    Interface Tabs
    Contributing
    License

Overview

The provided Python script offers a Gradio-powered interface to generate images using the Stable Diffusion model. The interface provides three tabs, each offering a different method of image generation:

    Webcam: Capture images using your webcam and apply the Stable Diffusion transformation.
    Upload: Upload an image from your local system and apply the transformation.
    Text: Input a prompt to guide the image transformation based on the provided text.

Users can also adjust the transformation parameters such as strength and guidance_scale to control the extent and style of the image transformation.
Dependencies

To run the script, you'll need the following dependencies:

    Python (>= 3.6)
    torch
    PIL (Pillow)
    gradio
    diffusers (StableDiffusionImg2ImgPipeline)
    RunwayML (for the Stable Diffusion model)

You can install the required dependencies using the following command:

pip install torch pillow gradio runway-diffusers

Usage

Clone this repository to your local machine:

gh repo clone etemkocaaslan/imageGen

    Install the required dependencies as mentioned in the Dependencies section.
    Run the Python script:

python app.py

    Access the Gradio interface through your web browser.

Contributing

Contributions to this project are welcome! If you would like to contribute enhancements or fix issues, please follow these steps:

    Fork the Repository: Fork this repository by clicking the "Fork" button at the top right.
    Create a Branch: Create a new branch for your feature or fix using a descriptive name, such as feature/new-feature or fix/issue-description.
    Make Changes: Make your changes in the new branch and commit them.
    Push Changes: Push the changes to your forked repository.
    Open a Pull Request: Open a pull request (PR) to the original repository, describing the changes you've made.

License

This project is licensed under the MIT License - see the LICENSE file for details
