import torch as th
from torchvision.transforms import ToPILImage

def detect_faces(images, mtcnn):
    # Konvertiere die Bilder in PIL-Bilder
    pil_images = [ToPILImage()(img) for img in images]

    # Erkenne Gesichter in den Bildern und erhalte die Bounding Boxes
    faces = []
    boxes = []
    for image in pil_images:
        image_boxes, _ = mtcnn.detect(image)
        if image_boxes is not None:
            image_faces = mtcnn(image)
            faces.append(th.unsqueeze(image_faces[0], 0))
            boxes.append(image_boxes[0])
        else:
            faces.append(None)
            boxes.append(None)

    return faces, boxes