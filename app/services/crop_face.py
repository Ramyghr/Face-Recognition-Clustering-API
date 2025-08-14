#crop_face.py
import numpy as np
def crop_faces(image, bounding_boxes):
    """
    Crop faces from an image based on bounding boxes.

    Args:
        image (ndarray): Input image.
        bounding_boxes (list): List of bounding boxes (x, y, w, h).

    Returns:
        list: List of cropped face images.
    """
    cropped_faces = []
    for box in bounding_boxes:
        x, y, w, h = box
        cropped_face = image[y:y+h, x:x+w]
        cropped_faces.append(cropped_face)
    return cropped_faces