import os
from PIL import ImageDraw
from torchvision.utils import make_grid, save_image
from torchvision.transforms import ToPILImage

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