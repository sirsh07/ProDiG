import os
import requests
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from peft import LoraConfig
p = "src/"
sys.path.append(p)
from einops import rearrange, repeat
from pipeline_aerofix import aerofixPipeline
from easydict import EasyDict as edict


from geometry import compute_epipolar_mask, compute_plucker_embed, get_mask_and_plucker

def make_1step_sched():
    noise_scheduler_1step = DDPMScheduler.from_pretrained("stabilityai/sd-turbo", subfolder="scheduler")
    noise_scheduler_1step.set_timesteps(1, device="cuda")
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.cuda()
    return noise_scheduler_1step


def my_vae_encoder_fwd(self, sample):
    sample = self.conv_in(sample)
    l_blocks = []
    # down
    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)
    # middle
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = l_blocks
    return sample


def my_vae_decoder_fwd(self, sample, latent_embeds=None):
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    # middle
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    if not self.ignore_skip:
        skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
        # up
        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
            # add skip
            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
    else:
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)
    # post-process
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample


def download_url(url, outf):
    if not os.path.exists(outf):
        print(f"Downloading checkpoint to {outf}")
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(outf, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")
        print(f"Downloaded successfully to {outf}")
    else:
        print(f"Skipping download, {outf} already exists")


def load_ckpt_from_state_dict(net_aerofix, optimizer, pretrained_path):
    sd = torch.load(pretrained_path, map_location="cpu")
    
    if "state_dict_vae" in sd:
        _sd_vae = net_aerofix.vae.state_dict()
        for k in sd["state_dict_vae"]:
            _sd_vae[k] = sd["state_dict_vae"][k]
        net_aerofix.vae.load_state_dict(_sd_vae)
    _sd_unet = net_aerofix.unet.state_dict()
    for k in sd["state_dict_unet"]:
        _sd_unet[k] = sd["state_dict_unet"][k]
    net_aerofix.unet.load_state_dict(_sd_unet)
        
    optimizer.load_state_dict(sd["optimizer"])
    
    return net_aerofix, optimizer


def save_ckpt(net_aerofix, optimizer, outf):
    sd = {}
    sd["vae_lora_target_modules"] = net_aerofix.target_modules_vae
    sd["rank_vae"] = net_aerofix.lora_rank_vae
    sd["state_dict_unet"] = net_aerofix.unet.state_dict()
    # sd["state_dict_vae"] = {k: v for k, v in net_aerofix.vae.state_dict().items() if "lora" in k or "skip" in k}
    sd["state_dict_vae"] = net_aerofix.vae.state_dict()
    
    sd["optimizer"] = optimizer.state_dict()   
    
    torch.save(sd, outf)
    
# def merge_vae_lora(vae, adapter_name="vae_skip", scale=1.0):
#     for name, module in vae.named_modules():
#         if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
#             delta_weight = scale * (module.lora_B[adapter_name] @ module.lora_A[adapter_name])
#             module.weight.data += delta_weight
#             # Optionally clean up
#             del module.lora_A
#             del module.lora_B
#     # Clean peft_config
#     if hasattr(vae, "peft_config"):
#         del vae.peft_config[adapter_name]
#     print(f"✅ Merged LoRA adapter '{adapter_name}' into VAE.")


import torch
import torch.nn as nn

def merge_vae_lora(module, adapter_name="vae_skip", scale=1.0):
    for name, submodule in module.named_modules():
        if hasattr(submodule, "lora_A") and hasattr(submodule, "lora_B"):
            # print(f"Merging LoRA in: {name} ({type(submodule).__name__})")
            
            # import pdb; pdb.set_trace()
            
            # TODO (Sirsh)-(07-08-2025-17:50): make it generalizable to a single lora type
            submodule.merge()
             
            # Get LoRA weights
            # lora_A = submodule.lora_A[adapter_name].weight
            # lora_B = submodule.lora_B[adapter_name].weight

            # if isinstance(submodule, nn.Linear):
            #     # Linear LoRA merge: W + scale * (B @ A)
            #     delta_weight = scale * torch.matmul(lora_B, lora_A)
            #     submodule.weight.data += delta_weight

            # elif isinstance(submodule, nn.Conv2d):
            #     # Conv2d LoRA merge: W + scale * (B * A)
            #     delta_weight = scale * (lora_B * lora_A)
            #     submodule.weight.data += delta_weight
                
            # elif "peft.tuners.lora.layer.Conv2d" in str(type(submodule)):
            #         # PEFT Conv2d LoRA merge
            #         base_weight = submodule.base_layer.weight.data
            #         delta_weight = scale * (lora_B * lora_A)
            #         submodule.base_layer.weight.data = base_weight + delta_weight

            # else:
            #     print(f"Unknown module type: {type(submodule)}. Skipping.")

            # Clean up LoRA adapters
            del submodule.lora_A
            del submodule.lora_B
    if hasattr(module, "peft_config"):
        del module.peft_config[adapter_name]
    print(f"Merged LoRA adapter '{adapter_name}' into VAE.")




class aerofix(torch.nn.Module):
    # def __init__(self, pretrained_name=None, pretrained_path=None, ckpt_folder="checkpoints", lora_rank_vae=4, mv_unet=False, timestep=999, train_unet=True, train_vae=True, add_vae=True, mv_unet_v=None, add_unet_lora=True):
    def __init__(self, pretrained_name=None, pretrained_path=None, ckpt_folder="checkpoints", lora_rank_vae=4, lora_rank_unet=4, mv_unet=False, timestep=999, train_vae=True, add_vae=True, mv_unet_v=None, finetune_unet="no", pose_embed=False):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()
        self.sched = make_1step_sched()
        
        self.finetune_unet = finetune_unet
        self.train_vae = train_vae

        # if pretrained_name is not None:
        #     vae = AutoencoderKL.from_pretrained(pretrained_name, subfolder="vae")
        # else:
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        # add the skip connection convs
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.ignore_skip = False
        
        
        if mv_unet:
            # raise NotImplementedError("Multi-view UNet is not implemented yet")
            if mv_unet_v is None or mv_unet_v == "none":
                from mv_unet import UNet2DConditionModel, enable_patch
            # elif mv_unet_v == "v1":
            #     from models_old.mv_unet_v1 import UNet2DConditionModel, enable_patch
            # elif mv_unet_v == "v2":
            #     from models.mv_unet import UNet2DConditionModel, enable_patch
            # elif mv_unet_v == "v3":
            #     from models_v3.mv_unet import UNet2DConditionModel, enable_patch
            elif mv_unet_v == "v4":
                from models_v4.mv_unet import UNet2DConditionModel
            # elif mv_unet_v == "v5":
            #     from models_v5.mv_unet import UNet2DConditionModel
            # elif mv_unet_v == "v6":
            #     from models_v6.mv_unet import UNet2DConditionModel
            # elif mv_unet_v == "v7":
            #     from models_v7.mv_unet import UNet2DConditionModel
        else:
            from diffusers import UNet2DConditionModel
            
        self.pose_embed = pose_embed
        if pose_embed:
            unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet", low_cpu_mem_usage=False, ignore_mismatched_sizes=True,time_cond_proj_dim=12)    
        
        else:
            unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet", low_cpu_mem_usage=False, ignore_mismatched_sizes=True)

        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            # import pdb; pdb.set_trace()
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)
            
        if pretrained_name is not None:
            # pipeline_unet = UNet2DConditionModel.from_pretrained(pretrained_name, subfolder="unet")
            # pipeline_vae = UNet2DConditionModel.from_pretrained(pretrained_name, subfolder="vae")
            
            # _sd_unet = unet.state_dict()
            # for k in pipeline_unet.state_dict():
            #     _sd_unet[k] = pipeline_unet.state_dict()[k]
            # unet.load_state_dict(_sd_unet)
            # import pdb; pdb.set_trace()

            pipeline = aerofixPipeline.from_pretrained(pretrained_name, trust_remote_code=True)
            # pipeline = aerofixPipeline.from_pretrained(pretrained_name, trust_remote_code=False)
            _sd_unet = unet.state_dict()
            for k in pipeline.unet.state_dict():
                _sd_unet[k] = pipeline.unet.state_dict()[k]
            unet.load_state_dict(_sd_unet)
            
            
            # if not self.train_unet:
                
            #     # NOTE (Sirsh)-(08-27-2025-19:41):
                
            #     if add_unet_lora:
            
            #         unet.requires_grad_(False)
            #         unet_lora_config = LoraConfig(r=lora_rank_vae,init_lora_weights="gaussian",target_modules=["to_k", "to_q", "to_v", "to_out.0"],)
            #         unet.add_adapter(unet_lora_config)
                    
            #     else:
                    
            #         unet.requires_grad_(False)
            #         # for n, _ap in unet.named_parameters():
            #         #     if "devoa" in n:
            #         #         # if "norm_devoa" not in n:
            #         #         # _ap.zero_()
            #         #         _ap.requires_grad = True
            
            
            if self.finetune_unet != "no":
                '''
                no: do not train unet
                full: train the whole unet
                lora: add lora layers to unet and train them
                attn: train only the attention layers --- q,k,v,out
                qkv: train only the q,k,v layers
                attn1: train only the first attention layers in the cross attention block --- q,k,v,out
                qkv1: train only the q,k,v layers in the first attention layers in the cross attention block --- q,k,v 
                '''
                
                
                print(f"Fine-tuning UNet with mode: {self.finetune_unet}")
                if self.finetune_unet == "full":
                    unet.requires_grad_(True)
                elif self.finetune_unet == "lora":
                    unet.requires_grad_(False)
                    # unet_lora_config = LoraConfig(r=lora_rank_unet,init_lora_weights="gaussian",target_modules=["to_k", "to_q", "to_v", "to_out.0"],)
                    # target_modules_unet =  ["to_k", "to_q", "to_v", "to_out.0"]
                    target_modules_unet = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out","proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"]
                    unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian",
                        target_modules=target_modules_unet
                    )
                    unet.add_adapter(unet_lora_config)
                    
                    self.lora_rank_unet = lora_rank_unet
                    self.target_modules_unet = target_modules_unet
                    
                    for n, p in unet.named_parameters():
                        if "plucker_proj" in n:
                            p.requires_grad = True
                        # if mv_unet_v == "v4" or mv_unet_v == "v3":
                        if pose_embed and "time" in n:
                            p.requires_grad = True
                        if "lora" in n:
                            p.requires_grad = True
                    
                elif self.finetune_unet == "attn":
                    unet.requires_grad_(False)
                    for n, p in unet.named_parameters():
                        if "attn" in n:
                            p.requires_grad = True
                        if "plucker_proj" in n:
                            p.requires_grad = True
                        # if mv_unet_v == "v4" or mv_unet_v == "v3":
                        if pose_embed and "time" in n:
                            p.requires_grad = True
                
                elif self.finetune_unet == "qkv":
                    unet.requires_grad_(False)
                    for n, p in unet.named_parameters():
                        if "to_q" in n or "to_k" in n or "to_v" in n:
                            p.requires_grad = True
                        if "plucker_proj" in n:
                            p.requires_grad = True
                        # if mv_unet_v == "v4" or mv_unet_v == "v3":
                        if pose_embed and "time" in n:
                            p.requires_grad = True
                        
                elif self.finetune_unet == "attn1":
                    unet.requires_grad_(False)
                    for n, p in unet.named_parameters():
                        if "attn1" in n:
                            p.requires_grad = True
                        if "plucker_proj" in n:
                            p.requires_grad = True
                        # if mv_unet_v == "v4" or mv_unet_v == "v3":
                        if pose_embed and "time" in n:
                            p.requires_grad = True
                elif self.finetune_unet == "qkv1":
                    unet.requires_grad_(False)
                    for n, p in unet.named_parameters():
                        if "attn1" in n:
                            if "to_q" in n or "to_k" in n or "to_v" in n:
                                p.requires_grad = True
                        if "plucker_proj" in n:
                            p.requires_grad = True
                        # if mv_unet_v == "v4" or mv_unet_v == "v3":
                        if pose_embed and "time" in n:
                            p.requires_grad = True
                            
                else:
                    raise ValueError(f"Unknown finetune_unet mode: {self.finetune_unet}")
            else:
                unet.requires_grad_(False)    
            
            # print(f"Trainable params: {[n for n,p in unet.named_parameters() if p.requires_grad]}")  
            # print(f"Non-Trainable params: {[n for n,p in unet.named_parameters() if not p.requires_grad]}")

            if add_vae:
                merge_vae_lora(pipeline.vae, adapter_name="vae_skip", scale=1.0)
            
                _sd_vae = vae.state_dict()
                for k in pipeline.vae.state_dict():
                    _sd_vae[k.replace(".base_layer", "")] = pipeline.vae.state_dict()[k] # for the lora layers
                vae.load_state_dict(_sd_vae)
                
                print("Initializing model with pretrained_name weights")
                
                target_modules_vae = []

                torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
                torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
                torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
                torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
                target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                    "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                    "to_k", "to_q", "to_v", "to_out.0",
                ]
                
                target_modules = []
                for id, (name, param) in enumerate(vae.named_modules()):
                    if 'decoder' in name and any(name.endswith(x) for x in target_modules_vae):
                        target_modules.append(name)
                target_modules_vae = target_modules
                vae.encoder.requires_grad_(False)

                vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",
                    target_modules=target_modules_vae)
                
                
                vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
                
                self.lora_rank_vae = lora_rank_vae
                self.target_modules_vae = target_modules_vae
                
            else:
            
                self.lora_rank_vae = pipeline.vae.peft_config["vae_skip"].r
                self.target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                                       "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4", 
                                       "to_k", "to_q", "to_v", "to_out.0",]
                
            # del pipeline
            # torch.cuda.empty_cache()
            if mv_unet_v not in ["v4","v5","v6","v7"]:
                enable_patch()
         
        elif pretrained_name is None and pretrained_path is None:
            print("Initializing model with random weights")
            target_modules_vae = []

            torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
            target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                "to_k", "to_q", "to_v", "to_out.0",
            ]
            
            target_modules = []
            for id, (name, param) in enumerate(vae.named_modules()):
                if 'decoder' in name and any(name.endswith(x) for x in target_modules_vae):
                    target_modules.append(name)
            target_modules_vae = target_modules
            vae.encoder.requires_grad_(False)

            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",
                target_modules=target_modules_vae)
            
            
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
                
            self.lora_rank_vae = lora_rank_vae
            self.target_modules_vae = target_modules_vae

        # unet.enable_xformers_memory_efficient_attention()
        unet.to("cuda")
        vae.to("cuda")

        self.unet, self.vae = unet, vae
        self.vae.decoder.gamma = 1
        self.timesteps = torch.tensor([timestep], device="cuda").long()
        self.text_encoder.requires_grad_(False)

        # import pdb; pdb.set_trace()
        # print number of trainable parameters
        print("="*50)
        print("After initialization:")
        print(f"Number of trainable parameters in UNet: {sum(p.numel() for p in unet.parameters() if p.requires_grad) / 1e6:.2f}M")
        print(f"Number of trainable parameters in VAE: {sum(p.numel() for p in vae.parameters() if p.requires_grad) / 1e6:.2f}M")
        print("="*50)
        
    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        
        if self.finetune_unet == "full":
            self.unet.requires_grad_(True)
        if self.finetune_unet == "lora":
            for n, p in self.unet.named_parameters():
                if "lora" in n:
                    p.requires_grad = True
                if "plucker_proj" in n:
                    p.requires_grad = True
                # if mv_unet_v == "v4" or mv_unet_v == "v3":
                if self.pose_embed and "time" in n:
                    p.requires_grad = True
            self.unet.conv_in.requires_grad_(True)
            

        if self.train_vae:
            for n, _p in self.vae.named_parameters():
                if "lora" in n:
                    _p.requires_grad = True
            self.vae.decoder.skip_conv_1.requires_grad_(True)
            self.vae.decoder.skip_conv_2.requires_grad_(True)
            self.vae.decoder.skip_conv_3.requires_grad_(True)
            self.vae.decoder.skip_conv_4.requires_grad_(True)
        
        
        print("="*50)
        print("After setting train mode:")
        print(f"Number of trainable parameters in UNet: {sum(p.numel() for p in self.unet.parameters() if p.requires_grad) / 1e6:.2f}M")
        print(f"Number of non-trainable parameters in UNet: {sum(p.numel() for p in self.unet.parameters() if not p.requires_grad) / 1e6:.2f}M")
        print(f"Number of trainable parameters in VAE: {sum(p.numel() for p in self.vae.parameters() if p.requires_grad) / 1e6:.2f}M")
        print(f"Number of non-trainable parameters in VAE: {sum(p.numel() for p in self.vae.parameters() if not p.requires_grad) / 1e6:.2f}M")
        print("="*50)
        
        # import pdb; pdb.set_trace()
        

    def forward(self, x, timesteps=None, prompt=None, prompt_tokens=None, pose_mask=None, pose_embed=None, plucker_embed=None):
        # either the prompt or the prompt_tokens should be provided
        assert (prompt is None) != (prompt_tokens is None), "Either prompt or prompt_tokens should be provided"
        assert (timesteps is None) != (self.timesteps is None), "Either timesteps or self.timesteps should be provided"
        
        if prompt is not None:
            # encode the text prompt
            caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0]

        num_views = x.shape[1]
        x = rearrange(x, 'b v c h w -> (b v) c h w')
        z = self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor 
        caption_enc = repeat(caption_enc, 'b n c -> (b v) n c', v=num_views)
        # import pdb; pdb.set_trace()
        unet_input = z
        # import pdb; pdb.set_trace()
        # NOTE (Sirsh)-(10-13-2025-19:54): debugging attention masking ---_>
        # attention_mask = torch.stack([torch.ones_like(unet_input[0]), torch.zeros_like(unet_input[1])])
        # attention_mask = attention_mask.to(unet_input.device)
        # model_pred = self.unet(unet_input, self.timesteps, encoder_hidden_states=caption_enc,attention_mask=attention_mask).sample
        # ----<
        # original code:
       
        if pose_mask is not None:
            model_pred = self.unet(unet_input, self.timesteps, encoder_hidden_states=caption_enc, attention_mask=pose_mask, timestep_cond=pose_embed, added_cond_kwargs={"plucker_emb":plucker_embed}).sample            
        else:        
            model_pred = self.unet(unet_input, self.timesteps, encoder_hidden_states=caption_enc, timestep_cond=pose_embed).sample
            


        z_denoised = self.sched.step(model_pred, self.timesteps, z, return_dict=True).prev_sample
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        z_denoised = z_denoised.to(model_pred.dtype)
        output_image = (self.vae.decode(z_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        output_image = rearrange(output_image, '(b v) c h w -> b v c h w', v=num_views)
        
        return output_image
    
    def sample(self, image, width, height, ref_image=None, timesteps=None, prompt=None, prompt_tokens=None, pose_mask=None, pose_embed=None, plucker_embed=None):
        input_width, input_height = image.size
        new_width = image.width - image.width % 8
        new_height = image.height - image.height % 8
        image = image.resize((new_width, new_height), Image.LANCZOS)
        
        T = transforms.Compose([
            transforms.Resize((height, width), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        if ref_image is None:
            x = T(image).unsqueeze(0).unsqueeze(0).cuda()
        else:
            ref_image = ref_image.resize((new_width, new_height), Image.LANCZOS)
            x = torch.stack([T(image), T(ref_image)], dim=0).unsqueeze(0).cuda()
        
        # output_image = self.forward(x, timesteps, prompt, prompt_tokens)[:, 0]
        
        weight_dtype = torch.bfloat16
        x = x.to(weight_dtype)
        
        if pose_mask is not None:
            output_image = self.forward(x, timesteps, prompt, prompt_tokens,pose_mask=pose_mask, pose_embed=pose_embed, plucker_embed=plucker_embed)[:, 0]            
        else:        
            output_image = self.forward(x, timesteps, prompt, prompt_tokens,pose_embed=pose_embed)[:, 0]
            
        output_image = output_image.float()
            
        output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
        output_pil = output_pil.resize((input_width, input_height), Image.LANCZOS)
        
        return output_pil

    def save_model(self, outf, optimizer):
        sd = {}
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k}
        
        sd["optimizer"] = optimizer.state_dict()
        
        torch.save(sd, outf)
