import os
import random
from PIL import Image
from PIL import ImageDraw
from torchvision.utils import make_grid, save_image
from torchvision.transforms import ToPILImage
from torchvision.transforms import ToTensor

def visualize_faces(args, batch_images, batch_boxes, step, quantile):
    draw_images = [img.clone() for img in batch_images]
    for img, box in zip(draw_images, batch_boxes):
        img = ToPILImage()(img)
        draw = ImageDraw.Draw(img)
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=(255, 0, 0), width=4)

    grid = make_grid(draw_images, nrow=8)
    if args.quantiles < 1:
        output_dir = os.path.join(args.output_dir, 'face_visualization')
    else:
        output_dir = os.path.join(args.output_dir, f'face_visualization_{quantile}')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"faces_step_{step}.png")
    save_image(grid, output_file)

def visualize_images(args, metadata):
    # Zufällige Prompt-Gruppen auswählen
    prompt_groups = metadata.groupby('prompt')
    selected_prompts = random.sample(list(prompt_groups.groups.keys()), min(args.num_prompt_groups, len(prompt_groups)))

    for quantile in range(args.quantiles):
        quantile_metadata = metadata[metadata['quantile'] == quantile]

        # Zufälliges Grid
        random_indices = random.sample(range(len(quantile_metadata)), args.images_per_grid)
        random_images = [Image.open(os.path.join(args.data_dir, f"{idx}.png")).convert("RGB") for idx in quantile_metadata.iloc[random_indices]['idx']]
        random_tensors = [ToTensor()(img) for img in random_images]
        random_grid = make_grid(random_tensors, nrow=args.grid_nrow)
        output_dir = os.path.join(args.output_dir, f'image_visualization_quantile_{quantile}')
        os.makedirs(output_dir, exist_ok=True)
        save_image(random_grid, os.path.join(output_dir, f"random_images.png"))

        # Gruppen-Grids
        for prompt in selected_prompts:
            group_metadata = quantile_metadata[quantile_metadata['prompt'] == prompt]
            if len(group_metadata) > 0:
                group_indices = random.sample(range(len(group_metadata)), min(args.images_per_grid, len(group_metadata)))
                group_images = [Image.open(os.path.join(args.data_dir, f"{idx}.png")).convert("RGB") for idx in group_metadata.iloc[group_indices]['idx']]
                group_tensors = [ToTensor()(img) for img in group_images]
                group_grid = make_grid(group_tensors, nrow=args.grid_nrow)
                save_image(group_grid, os.path.join(output_dir, f"prompt_{prompt}.png"))