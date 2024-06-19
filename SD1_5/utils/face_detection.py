import torch as th
from torchvision.transforms import ToPILImage

def detect_faces(images, mtcnn):
    # Konvertiere die Bilder in PIL-Bilder
    if isinstance(images[0], th.Tensor):
        pil_images = [ToPILImage()(img) for img in images]
    else:
        pil_images = images

    # Erkenne Gesichter in den Bildern und erhalte die Bounding Boxes
    faces = []
    boxes = []
    for image in pil_images:
        image_boxes, _ = mtcnn.detect(image)
        if image_boxes is not None:
            for box in image_boxes:
                x1, y1, x2, y2 = box.astype(int)
                face_img = image.crop((x1, y1, x2, y2))
                face = mtcnn(face_img, return_prob=False)
                faces.append(face)
                boxes.append(box)
        else:
            faces.append(None)
            boxes.append(None)

    return faces, boxes