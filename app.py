import gradio as gr

import cv2
import torch
import numpy as np
import os
from control_cogvideox.cogvideox_transformer_3d import CogVideoXTransformer3DModel
from control_cogvideox.controlnet_cogvideox_transformer_3d import ControlCogVideoXTransformer3DModel
from pipeline_cogvideox_controlnet_5b_i2v_instruction2 import ControlCogVideoXPipeline
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLCogVideoX
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.schedulers import CogVideoXDDIMScheduler

from omegaconf import OmegaConf
from transformers import T5EncoderModel
from einops import rearrange
import decord
from typing import List
from tqdm import tqdm

import PIL
import torch.nn.functional as F
from torchvision import transforms

from huggingface_hub import snapshot_download

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


print("Download successfully!")

def get_prompt(file:str):
    with open(file,'r') as f:
        a=f.readlines()
    return a #a[0]:positive prompt, a[1] negative prompt
def unwarp_model(state_dict):
    new_state_dict = {}
    for key in state_dict:
        new_state_dict[key.split('module.')[1]] = state_dict[key]
    return new_state_dict

def init_pipe():

    i2v=True

    if i2v:
        key = "i2v"
    else:
        key = "t2v"
    noise_scheduler = CogVideoXDDIMScheduler(
        **OmegaConf.to_container(
            OmegaConf.load(f"./cogvideox-5b-{key}/scheduler/scheduler_config.json")
        )
    )

    text_encoder = T5EncoderModel.from_pretrained(f"./cogvideox-5b-{key}/", subfolder="text_encoder", torch_dtype=torch.float16)
    vae = AutoencoderKLCogVideoX.from_pretrained(f"./cogvideox-5b-{key}/", subfolder="vae", torch_dtype=torch.float16)
    tokenizer = T5Tokenizer.from_pretrained(f"./cogvideox-5b-{key}/tokenizer", torch_dtype=torch.float16)


    config = OmegaConf.to_container(
        OmegaConf.load(f"./cogvideox-5b-{key}/transformer/config.json")
    )
    if i2v:
        config["in_channels"] = 32
    else:
        config["in_channels"] = 16
    transformer = CogVideoXTransformer3DModel(**config)

    control_config = OmegaConf.to_container(
        OmegaConf.load(f"./cogvideox-5b-{key}/transformer/config.json")
    )
    if i2v:
        control_config["in_channels"] = 32
    else:
        control_config["in_channels"] = 16
    control_config['num_layers'] = 6
    control_config['control_in_channels'] = 16
    controlnet_transformer = ControlCogVideoXTransformer3DModel(**control_config)

    all_state_dicts = torch.load("./senorita-2m/models_half/ff_controlnet_half.pth", map_location="cpu",weights_only=True)
    transformer_state_dict = unwarp_model(all_state_dicts["transformer_state_dict"])
    controlnet_transformer_state_dict = unwarp_model(all_state_dicts["controlnet_transformer_state_dict"])

    transformer.load_state_dict(transformer_state_dict, strict=True)
    controlnet_transformer.load_state_dict(controlnet_transformer_state_dict, strict=True)

    transformer = transformer.half()
    controlnet_transformer = controlnet_transformer.half()

    vae = vae.eval()
    text_encoder = text_encoder.eval()
    transformer = transformer.eval()
    controlnet_transformer = controlnet_transformer.eval()

    pipe = ControlCogVideoXPipeline(tokenizer,
            text_encoder,
            vae,
            transformer,
            noise_scheduler,
            controlnet_transformer,
    )

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload()

    return pipe
    

def inference(source_images, 
        target_images, 
        text_prompt, negative_prompt, 
        pipe, vae, guidance_scale, 
        h, w, random_seed)->List[PIL.Image.Image]:
    torch.manual_seed(random_seed)
    
    pipe.vae.to(DEVICE)
    pipe.transformer.to(DEVICE)
    pipe.controlnet_transformer.to(DEVICE)

    source_pixel_values = source_images/127.5 - 1.0
    source_pixel_values = source_pixel_values.to(torch.float16).to(DEVICE)
    if target_images is not None:
        target_pixel_values = target_images/127.5 - 1.0
        target_pixel_values = target_pixel_values.to(torch.float16).to(DEVICE)
    bsz,f,h,w,c = source_pixel_values.shape

    with torch.no_grad():
        source_pixel_values = rearrange(source_pixel_values, "b f w h c -> b c f w h")
        source_latents = vae.encode(source_pixel_values).latent_dist.sample()
        source_latents = source_latents.to(torch.float16)
        source_latents = source_latents * vae.config.scaling_factor
        source_latents = rearrange(source_latents, "b c f h w -> b f c h w")

        if target_images is not None:
            target_pixel_values = rearrange(target_pixel_values, "b f w h c -> b c f w h")
            images = target_pixel_values[:,:,:1,...]
            image_latents = vae.encode(images).latent_dist.sample()
            image_latents = image_latents.to(torch.float16)
            image_latents = image_latents * vae.config.scaling_factor
            image_latents = rearrange(image_latents, "b c f h w -> b f c h w")
            image_latents = torch.cat([image_latents, torch.zeros_like(source_latents)[:,1:]],dim=1)
            latents = torch.cat([image_latents, source_latents], dim=2)
        else:
            image_latents = None
            latents = source_latents

    video = pipe(
        prompt = text_prompt,
        negative_prompt = negative_prompt,
        video_condition = source_latents, # input to controlnet
        video_condition2 = image_latents, # concat with latents
        height = h,
        width = w,
        num_frames = f,
        num_inference_steps = 30,
        interval = 6,
        guidance_scale = guidance_scale,
        generator = torch.Generator(device=DEVICE).manual_seed(random_seed)
    ).frames[0]

    return video

def process_video(video_file, image_file, positive_prompt, negative_prompt, guidance, random_seed, choice, progress=gr.Progress(track_tqdm=True))->str:
    if choice==33:
        video_shard=1
    elif choice==65:
        video_shard=2
    
    pipe=PIPE

    h = 448
    w = 768
    frames_per_shard=33

    #get image
    image = cv2.imread(image_file)
    resized_image = cv2.resize(image, (768, 448))
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    image=torch.from_numpy(resized_image)
    #get mp4
    vr = decord.VideoReader(video_file)
    frames = vr.get_batch(list(range(33))).asnumpy()
    _,src_h,src_w,_=frames.shape
    resized_frames = [cv2.resize(frame, (768, 448)) for frame in frames]
    images=torch.from_numpy(np.array(resized_frames))

    target_path="outputvideo"
    source_images = images[None,...]
    target_images = image[None,None,...]
    
    video:List[PIL.Image.Image]=[]

    for i in progress.tqdm(range(video_shard)):
        if i>0: #first frame guidence
            first_frame=transforms.ToTensor()(video[-1])
            first_frame = first_frame*255.0
            first_frame = rearrange(first_frame,"c w h -> w h c")
            source_images=source_images
            target_images=first_frame[None,None,...]
            
        video+=inference(source_images, \
            target_images, positive_prompt, \
            negative_prompt, pipe, pipe.vae, \
            guidance, \
            h, w, random_seed)
        i+=1

    video=[image.resize((int(src_w/src_h*448),448))for image in video]

    os.makedirs(f"./{target_path}", exist_ok=True)
    output_path:str=f"./{target_path}/output_{video_file[-5]}.mp4"
    export_to_video(video, output_path, fps=8)
    return output_path
    

PIPE=init_pipe()


with gr.Blocks() as demo:
    gr.Markdown(
        """
        
        # Señorita-2M: A High-Quality Instruction-based Dataset for General Video Editing by Video Specialists
            
        [Paper](https://arxiv.org/abs/2502.06734) | [Code](https://127.0.0.1:7860) | [Huggingface](https://127.0.0.1:7860)
        <small>This UI is made by [PengWeixuanSZU](https://huggingface.co/PengWeixuanSZU).</small>
        """
    )
    
    with gr.Row():
        video_input = gr.Video(label="Video input")
        image_input = gr.Image(type="filepath", label="First frame guidence")
    with gr.Row():
        with gr.Column():
            positive_prompt = gr.Textbox(label="Positive prompt",value="")
            negative_prompt = gr.Textbox(label="Negative prompt",value="")
            seed = gr.Slider(minimum=0, maximum=2147483647, step=1, value=0, label="Seed")
            guidance_slider = gr.Slider(minimum=1, maximum=10, value=4, label="Guidance")
            choice=gr.Radio(choices=[33,65],label="Frame number",value=33)
        with gr.Column():
            video_output = gr.Video(label="Video output")
            
    with gr.Row():
        submit_button = gr.Button("Generate")
        submit_button.click(fn=process_video, inputs=[video_input, image_input, positive_prompt, negative_prompt, guidance_slider, seed, choice], outputs=video_output)
    with gr.Row():
        gr.Examples(
            [
                ["assets/0.mp4","assets/0_edit.png",get_prompt("assets/0.txt")[0],get_prompt("assets/0.txt")[1],4,0,33],
                ["assets/1.mp4","assets/1_edit.png",get_prompt("assets/1.txt")[0],get_prompt("assets/1.txt")[1],4,0,33],
                ["assets/2.mp4","assets/2_edit.png",get_prompt("assets/2.txt")[0],get_prompt("assets/2.txt")[1],4,0,33],
                ["assets/3.mp4","assets/3_edit.png",get_prompt("assets/3.txt")[0],get_prompt("assets/3.txt")[1],4,0,33],
                ["assets/4.mp4","assets/4_edit.png",get_prompt("assets/4.txt")[0],get_prompt("assets/4.txt")[1],4,0,33],
                ["assets/5.mp4","assets/5_edit.png",get_prompt("assets/5.txt")[0],get_prompt("assets/5.txt")[1],4,0,33],
                ["assets/6.mp4","assets/6_edit.png",get_prompt("assets/6.txt")[0],get_prompt("assets/6.txt")[1],4,0,33],
                ["assets/7.mp4","assets/7_edit.png",get_prompt("assets/7.txt")[0],get_prompt("assets/7.txt")[1],4,0,33],
                ["assets/8.mp4","assets/8_edit.png",get_prompt("assets/8.txt")[0],get_prompt("assets/8.txt")[1],4,0,33]
            ],
            inputs=[video_input, image_input, positive_prompt, negative_prompt, guidance_slider, seed, choice],
            outputs=video_output,
            fn=process_video,
            cache_examples=False
        )
    
demo.queue().launch()
