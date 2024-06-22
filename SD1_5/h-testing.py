import torch as th 
from accelerate import Accelerator

from utils.custom_pipe import HDiffusionPipeline

def main():
    use_fp16 = False

    accelerator = Accelerator()
  
    model_id = "SG161222/Realistic_Vision_V2.0"
    pipe = HDiffusionPipeline.from_pretrained(model_id, torch_dtype=th.float16 if use_fp16 else th.float32)
    
    # Move the model to device before preparing
    pipe = pipe.to(accelerator.device)
    pipe = accelerator.prepare(pipe)
    pipe.init_classifier()
    
    images, h_vects = pipe("A photo of the face of a female firefighter", num_inference_steps=50, guidance_scale=7.5, return_dict=False)
    images[0].save("test.png")
    
if __name__ == "__main__":
    main()