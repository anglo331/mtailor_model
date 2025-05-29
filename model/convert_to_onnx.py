from torch import nn 
from torch.onnx import export
from pytorch_model import *
from PIL import Image

def convert_to_onnx(model, input_shape:int = 3*224*224 , output_path='./model.onnx'):
    """
    Convert a PyTorch model to ONNX format.

    Args:
        model (nn.Module): The PyTorch model to convert.
        input_shape (tuple): The shape of the input tensor (excluding batch size).
        output_path (str): The path where the ONNX model will be saved.
    """
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)  # Create a dummy input tensor with the shape (batch_size, channels, height, width)

    # Export the model to ONNX format
    export(model, dummy_input, output_path, export_params=True, opset_version=11, do_constant_folding=True)
    
    print(f"Model has been converted to ONNX format and saved at {output_path}")


if __name__ == "__main__":
    model = Classifier(BasicBlock, [2, 2, 2, 2])
    model.load_state_dict(torch.load('mtailor_model/model/pytorch_model_weights.pth', weights_only=True))
    model.eval()

    inp = Image.open('mtailor_model/src/n01440764_tench.jpeg')

    inp = model.preprocess_numpy(inp).unsqueeze(0) 

    res = model.forward(inp)
    print(torch.argmax(res))

    convert_to_onnx(model, output_path='mtailor_model/model/mtailor_model.onnx')




