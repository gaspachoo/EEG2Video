'''
Description: 
Author: Zhou Tianyi
LastEditTime: 2025-04-11 12:10:33
LastEditors:  
'''
from pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline
from models.unet import UNet3DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from tuneavideo.util import save_videos_grid,ddim_inversion
import torch
from models.train_semantic_predictor import CLIP
import numpy as np
from einops import rearrange
from sklearn import preprocessing
import random
import math
model = None
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
import os
from diffusers.schedulers import (
    DDPMScheduler,
    DDIMScheduler,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device:", device)
def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        ## needs to be False to use conv3D
        print('Note: not using cudnn.deterministic')
seed_everything(114514)


eeg = torch.from_numpy(np.load('./data/EEG/sub1.npy')).to("cpu")

negative = eeg.mean(dim=0)

pretrained_model_path = "./stable-diffusion-v1-4" #"Zhoutianyi/huggingface/stable-diffusion-v1-4"
my_model_path = "./outputs/40_classes_200_epoch"

unet = UNet3DConditionModel.from_pretrained(my_model_path, subfolder='unet', torch_dtype=torch.float16).to(device)
pipe = TuneAVideoPipeline.from_pretrained(pretrained_model_path ,unet=unet, torch_dtype=torch.float16).to(device)
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_slicing()


# Ablation, inference w/o Seq2Seq and w/o DANA
woSeq2Seq = True
woDANA = False

if woSeq2Seq:
    latents = np.load('./outputs/latent_out_block7_40_classes.npy')
    latents = torch.from_numpy(latents).half()
    latents = rearrange(latents, 'a b c d e -> a c b d e')
    latents = latents.to(device)

if woDANA:
    latents_add_noise = torch.load('DANA/40_classes_latent_add_noise.pt')
    latents_add_noise = latents_add_noise.half()
    latents_add_noise = rearrange(latents_add_noise, 'a b c d e -> a c b d e')
    latents_add_noise = latents_add_noise.to(device)

for i in range(len(eeg)):
    if woSeq2Seq:
        video = pipe(model, eeg[i:i+1,...],negative_eeg=negative, latents=None, video_length=6, height=288, width=512, num_inference_steps=100, guidance_scale=12.5).videos
        savename = f'./EEG2Video_New/checkpoints/40_Classes_woSeq2Seq/'
        import os
        os.makedirs(savename, exist_ok=True)
    elif woDANA:
        video = pipe(model, eeg[i:i+1,...],negative_eeg=negative, latents=latents[i:i+1,...], video_length=6, height=288, width=512, num_inference_steps=100, guidance_scale=12.5).videos
        savename = f'./EEG2Video_New/checkpoints/40_Classes_woDANA/'
        import os
        os.makedirs(savename, exist_ok=True)
    else:
        video = pipe(model, eeg[i:i+1,...],negative_eeg=negative, latents=latents_add_noise[i:i+1,...], video_length=6, height=288, width=512, num_inference_steps=100, guidance_scale=12.5).videos
        savename = f'./EEG2Video_New/checkpoints/40_Classes_Fullmodel/'
        import os
        os.makedirs(savename, exist_ok=True)
    save_videos_grid(video, f"./{savename}/{i}.gif")
 