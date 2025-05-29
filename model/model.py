import onnxruntime as ort
from .pytorch_model import Classifier
from PIL import Image
from torchvision import transforms
import numpy as np

def create_session(model_path: str, providers: list = None) -> ort.InferenceSession:
    """
    Create an ONNX Runtime session for the specified model.

    Args:
        model_path (str): Path to the ONNX model file.
        providers (list, optional): List of execution providers to use. Defaults to None.

    Returns:
        ort.InferenceSession: An ONNX Runtime inference session.
    """
    if providers is None:
        providers = ['CPUExecutionProvider']

    session = ort.InferenceSession(model_path, providers=providers)
    return session


def predict(session: ort.InferenceSession, input_data: list) -> list:
    """
    Run inference on the ONNX model with the provided input data.

    Args:
        session (ort.InferenceSession): The ONNX Runtime session.
        input_data (list): Input data for the model.

    Returns:
        list: Model predictions.
    """
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    result = session.run([output_name], {input_name: input_data})
    
    return result[0]


def preprocess_input(image_path: str) -> list:
    """
    Preprocess the input image to be compatible with the model.
    """

    img = Image.open(image_path).convert('RGB')

    resize = transforms.Resize((224, 224))   #must same as here
    crop = transforms.CenterCrop((224, 224))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img = resize(img)
    img = crop(img)
    img = to_tensor(img)
    img = normalize(img)

    return img.unsqueeze(0).numpy().astype(np.float32).tolist() # Add batch dimension and convert to numpy array