import argparse
import os
import numpy as np
import torch as th
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import torch
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from guided_diffusion.script_util import (
    classifier_defaults,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)

from utils.custom_pipe import ClassifierGuidedStableDiffusionPipeline

script_dir = os.path.dirname(os.path.abspath(__file__))

def main():
    args = create_argparser().parse_args()

    args.classifier_path = os.path.join(script_dir, args.classifier_path)

    if args.seed is not None:
        th.manual_seed(args.seed)
        np.random.seed(args.seed)

    accelerator = Accelerator()

    pipeline = ClassifierGuidedStableDiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=th.float16 if accelerator.device.type == "cuda" else th.float32,
    )
    pipeline.to(accelerator.device)

    if args.use_fp16:
        pipeline.unet.half()
        pipeline.text_encoder.half()

    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(th.load(args.classifier_path, map_location="cpu"))
    classifier.to(accelerator.device)
    if args.use_fp16:
        classifier.half()
    classifier.eval()

    if args.feature_extractor:
        f_extractor = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
        f_extractor.load_state_dict(th.load(args.feature_extractor, map_location="cpu"))
        f_extractor.to(accelerator.device)
        if args.use_fp16:
            f_extractor.half()
        f_extractor.eval()
    else:
        f_extractor = None

    images_per_prompt = pipeline(
        args.prompts,
        classifier=classifier,
        classifier_scale=args.classifier_scale,
        f_extractor=f_extractor,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=th.manual_seed(args.seed) if args.seed is not None else None,
        num_images_per_prompt=args.num_images_per_prompt,
        output_type="numpy",
    )

    if accelerator.is_main_process:
        for prompt_idx, images in enumerate(images_per_prompt):
            grid = make_grid(torch.tensor(images), nrow=8)
            save_image(grid, os.path.join(args.output_dir, f"prompt_{prompt_idx}.png"))

    accelerator.wait_for_everyone()
    print("Sampling complete")

def create_argparser():
    defaults = dict(
        prompts=["a photo of a cat", "a photo of a dog"],
        num_images_per_prompt=64,
        model_path="SG161222/Realistic_Vision_V2.0",
        classifier_path="model_5.pt",
        classifier_scale=2.0,
        seed=None,
        num_quantiles=4,
        feature_extractor=None,
        use_fp16=True,
        output_dir="output",
        num_inference_steps=50,
        guidance_scale=7.5,
    )
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()