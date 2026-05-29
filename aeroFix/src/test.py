from pipeline_aerofix import aerofixPipeline
from diffusers.utils import load_image
import argparse


def parse_args():
    
    parser = argparse.ArgumentParser(description="Test aerofix3d pipeline")
    parser.add_argument("--image_path", type=str, help="Path to the input image")
    parser.add_argument("--ref_image_path", type=str, help="Path to the reference image", default=None)
    parser.add_argument("--output_path", type=str, help="Path to save the output image")
    parser.add_argument("--ckpts", type=str, default="nvidia/aerofix_ref", help="Height of the input image")
    
    return parser.parse_args()

def main():
    
    
    args = parse_args()
    
    
    image = load_image(args.image_path)
    prompt = "remove degradation"
    output_path = args.output_path
    
    
    if args.ref_image_path:
        ref_image = load_image(args.ref_image_path)
        pipeline = aerofixPipeline.from_pretrained(args.ckpts, trust_remote_code=True)
        pipeline.to("cuda")
        output_image = pipeline(prompt, image=image, ref_image=ref_image, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]

    
    else:
        pipeline = aerofixPipeline.from_pretrained(args.ckpts, trust_remote_code=True)
        pipeline.to("cuda")
        output_image = pipeline(prompt, image=image, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
    
    output_image.save(output_path)


if __name__ == "__main__":
    main()
    
    


