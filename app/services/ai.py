import numpy as np
import onnxruntime as rt
from PIL import Image
import cv2
from app.services.resnet import get_resnet_embedding
import logging

logger = logging.getLogger(__name__)

def resize_face(image: np.ndarray, size=(224, 224)):
    """Resize an image to expected input shape keeping it as a NumPy array."""
    if image.ndim == 3 and image.shape[2] == 3:
        image = Image.fromarray(image.astype(np.uint8))
    else:
        raise ValueError("Expected image with shape (H, W, 3)")
    
    image = image.resize(size, Image.BILINEAR)
    image = np.array(image)
    return image.astype(np.float32)

def preprocess_face(image: np.ndarray, model: str):
    """Preprocess face for different model types"""
    if model == "VGG":
        # Convert RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Convert to float32 and subtract mean values for BGR channels
        image = image.astype(np.float32)
        mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
        image -= mean
        return image
    elif model == "RESNET":
        # Convert to float32 and scale to [-1, 1]
        image = image.astype(np.float32)
        image = (image - 127.5) / 128.0
        # Transpose to (channels, height, width)
        image = image.transpose((2, 0, 1))
        return image
    elif model == "FACENET":
        # Standardization for FaceNet
        mean = np.mean(image)
        std = np.std(image)
        std_adj = np.maximum(std, 1.0/np.sqrt(image.size))
        image = (image - mean) / std_adj
        return image
    return image

# Global singleton session
FACE_SESSION = None
FACE_INPUT_NAME = None
FACE_OUTPUT_NAME = None

async def init_face_model(model_path: str):
    """Initialize the face recognition model"""
    global FACE_SESSION, FACE_INPUT_NAME, FACE_OUTPUT_NAME
    try:
        FACE_SESSION = rt.InferenceSession(
            model_path, 
            providers=["CPUExecutionProvider"]
        )
        FACE_INPUT_NAME = FACE_SESSION.get_inputs()[0].name
        FACE_OUTPUT_NAME = FACE_SESSION.get_outputs()[0].name
        logger.info(f"Face model initialized: {model_path}")
    except Exception as e:
        logger.error(f"Failed to initialize face model: {e}")
        raise

def infer_face(model_path, image: np.ndarray):
    """Run inference on face image"""
    global FACE_SESSION, FACE_INPUT_NAME, FACE_OUTPUT_NAME
    
    if FACE_SESSION is None:
        raise RuntimeError("Face model not initialized. Call init_face_model() first.")
    
    # Ensure input is float32 and has batch dimension
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    if image.ndim == 3:
        image = np.expand_dims(image, 0)
    
    try:
        result = FACE_SESSION.run(
            [FACE_OUTPUT_NAME],
            {FACE_INPUT_NAME: image}
        )[0][0]  # Remove batch dim from output
        return result
    except Exception as e:
        logger.warning(f"Face inference failed: {e}")
        return None

def get_base_embedding(face_img, model_path, model_name):
    """
    Fast embedding generation without augmentation for speed
    """
    try:
        # Resize and preprocess
        size = (224, 224) if model_name != "FACENET" else (160, 160)
        resized = resize_face(face_img, size)
        preprocessed = preprocess_face(resized, model_name)
        
        if preprocessed is None:
            return None
            
        # Run inference
        if model_name == "FACENET":
            return infer_face(model_path, preprocessed)
        else:
            return get_resnet_embedding(model_path, preprocessed)
    except Exception as e:
        logger.warning(f"Embedding generation failed: {e}")
        return None

def augment_face(face):
    """Generate augmented versions of a face (used for high-quality mode)"""
    augmented = [face]
    h, w = face.shape[:2]
    
    # Only essential augmentations for performance
    try:
        # Horizontal flip
        augmented.append(cv2.flip(face, 1))
        
        # Small rotations
        center = (w // 2, h // 2)
        for angle in [-10, 10]:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(face, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            augmented.append(rotated)
        
        # Brightness adjustment
        augmented.append(cv2.convertScaleAbs(face, alpha=1.1, beta=5))
        
    except Exception as e:
        logger.warning(f"Face augmentation failed: {e}")
    
    return augmented

def get_augmented_embedding(face_img, model_path, model_name="FACENET"):
    """Get robust embedding using augmentation (slower but more accurate)"""
    # Get base embedding first
    base_embed = get_base_embedding(face_img, model_path, model_name)
    if base_embed is None:
        return None
    
    # Generate augmented versions
    try:
        augmented = augment_face(face_img)
        embeddings = [base_embed]
        
        for aug_face in augmented[:4]:  # Limit to 4 augmentations for speed
            embed = get_base_embedding(aug_face, model_path, model_name)
            if embed is not None:
                embeddings.append(embed)
        
        # Return mean embedding
        if len(embeddings) > 1:
            return np.mean(embeddings, axis=0)
        else:
            return base_embed
            
    except Exception as e:
        logger.warning(f"Augmented embedding failed: {e}")
        return base_embed

def inference_pipeline(image, model_path, model_name, size=(160, 160), use_augmentation=False):
    """
    Main embedding pipeline with optional augmentation
    
    Args:
        image: Face image as numpy array
        model_path: Path to the model file
        model_name: Model name (FACENET, VGG, RESNET)
        size: Target size tuple
        use_augmentation: Whether to use augmentation (slower but potentially more accurate)
    """
    if use_augmentation:
        return get_augmented_embedding(image, model_path, model_name)
    else:
        return get_base_embedding(image, model_path, model_name)

def get_fast_embedding(face_img, model_path, model_name="FACENET"):
    """
    Ultra-fast embedding without any augmentation
    """
    return get_base_embedding(face_img, model_path, model_name)