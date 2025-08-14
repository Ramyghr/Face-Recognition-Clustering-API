import numpy as np
import onnxruntime as rt
from PIL import Image

def get_resnet_embedding(model_path, image: np.ndarray):
    session = rt.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Ensure input is float32 and has batch dimension
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    if image.ndim == 3:
        image = np.expand_dims(image, 0)  # Add batch dim

    outputs = session.run([output_name], {input_name: image})
    return outputs[0][0]  # Remove batch dim from output

def resize_for_resnet(image: np.ndarray, size=(224, 224)) -> np.ndarray:
    """Resize an image to ResNet expected input shape (224x224), keeping it as a NumPy array."""
    if image.ndim == 3 and image.shape[2] == 3:
        image = Image.fromarray(image.astype(np.uint8))
    else:
        raise ValueError("Expected image with shape (H, W, 3)")
    
    image = image.resize(size, Image.BILINEAR)
    image = np.array(image)

    # Convert to CHW and add batch dimension
    # image = image.transpose(2, 0, 1)  # HWC -> CHW
    image = np.expand_dims(image, 0)  # Add batch dimension

    return image.astype(np.float32)