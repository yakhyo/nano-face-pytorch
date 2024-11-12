import os
import argparse
import torch

from models import RetinaFace, SlimFace, RFB
from config import get_config


def parse_arguments():
    parser = argparse.ArgumentParser(description='ONNX Export')

    parser.add_argument(
        '-w', '--weights',
        default='./weights/last.pth',
        type=str,
        help='Trained state_dict file path to open'
    )
    parser.add_argument(
        '--network',
        type=str,
        default='retinaface',
        choices=['retinaface', 'slim', 'rfb'],
        help='Select a model architecture for face detection'
    )

    return parser.parse_args()


@torch.no_grad()
def onnx_export(params):
    # Get model configuration
    cfg = get_config(params.network)
    if cfg is None:
        raise KeyError(f"Config file for {params.network} not found!")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    if params.network == "retinaface":
        model = RetinaFace(cfg=cfg)
    elif params.network == "slim":
        model = SlimFace(cfg=cfg)
    elif params.network == "rfb":
        model = RFB(cfg=cfg)
    else:
        raise NameError("Please choose existing face detection method!")

    model.to(device)

    # Load weights
    state_dict = torch.load(params.weights, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print("Model loaded successfully!")

    # Set model to evaluation mode
    model.eval()

    # Generate output filename
    fname = os.path.splitext(os.path.basename(params.weights))[0]
    onnx_model = f'{fname}.onnx'
    print(f"==> Exporting model to ONNX format at '{onnx_model}'")

    # Create dummy input (batch_size=1, channels=3, height=640, width=640)
    x = torch.randn(1, 3, 640, 640).to(device)

    # Export model to ONNX
    torch.onnx.export(
        model,                # PyTorch Model
        x,                    # Model input
        onnx_model,          # Output file path
        export_params=True,   # Store the trained parameter weights inside the model file
        opset_version=11,    # ONNX version to export the model to
        do_constant_folding=True,  # Whether to execute constant folding for optimization
        input_names=['input'],     # Model's input names
        output_names=['loc', 'conf', 'landmarks'],  # Model's output names
        dynamic_axes={
            'input': {
                0: 'batch_size',
                2: 'height',
                3: 'width'
            },
            'loc': {0: 'batch_size'},      # Location output
            'conf': {0: 'batch_size'},     # Confidence output
            'landmarks': {0: 'batch_size'}  # Landmarks output
        }
    )

    print(f"Model exported successfully to {onnx_model}")


if __name__ == '__main__':
    args = parse_arguments()
    onnx_export(args)
