# ProDiG
Official Repository of CVPR26-Findings Paper "ProDiG: Progressive Diffusion-Guided Gaussian Splatting for Aerial to Ground Reconstruction".

```
accelerate launch --mixed_precision=bf16 src/train_aerofix.py \
  --output_dir="./outputs/" \
  --dataset_path="data.json" \
  --max_train_steps 10000 \
  --resolution 512 \
  --learning_rate 2e-5 \
  --train_batch_size 1 \
  --dataloader_num_workers 8 \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps 10000 \
  --eval_freq 1000 \
  --viz_freq 1000 \
  --lambda_lpips 1.0 \
  --lambda_l2 1.0 \
  --lambda_gram 1.0 \
  --gram_loss_warmup_steps 6000 \
  --report_to "tensorboard" \
  --tracker_project_name "aerofix" \
  --tracker_run_name "train" \
  --timestep 199 \
  --mv_unet \
  --pretrained_model_name_or_path "nvidia/difix_ref" \
  --finetune_unet="lora" \
  --pose_embed \
  --plucker_embed \
  --mv_unet_v "v4" 
```
