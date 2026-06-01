from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

import torch
import torch.nn.functional as F

from vgg_face_model import VGGFace

import cv2

"""
Runs inference with the VGG-Face model.

This code was informed by the following open-source PyTorch implementation:

    https://github.com/prlz77/vgg-face.pytorch/blob/master/models/vgg_face.py
"""
def preprocess_image(image_path: str, meta: dict) -> torch.Tensor:
    image = cv2.imread(image_path)
    h, w = meta['image_size'][0], meta['image_size'][1]
    image = cv2.resize(image, (w, h))
    
    # (H, W, 3) -> (3, H, W)
    tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

    mean = torch.tensor(meta['mean'], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(meta['std'], dtype=torch.float32).view(3, 1, 1)
    tensor = (tensor - mean) / std

    return tensor.unsqueeze(0)

def run_model(weights_path: str, image_path: str):
    print(f"Loading model from: {weights_path}")
    model = VGGFace()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    
    image = preprocess_image(image_path, model.meta)

    with torch.no_grad():
        logits = model(image)
        probs = F.softmax(logits, dim=1)

    # Top-5 predictions
    top5_probs, top5_indices = torch.topk(probs, 5, dim=1)
    print("\nTop-5 predictions:")
    for i in range(5):
        print(f"Class {top5_indices[0, i].item():4d}  "
              f"probability: {top5_probs[0, i].item():.6f}")

def build_arguments():
    arg_structure = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
        description="Verify converted VGG-Face model"
    )
    arg_structure.add_argument(
        "--weights",
        type=str,
        default="vgg_face.pth",
    )
    arg_structure.add_argument(
        "--image",
        type=str,
        default="vgg_face_torch/ak.png",
    )
    return arg_structure

def runCommLineIntf():
    arg_structure = build_arguments()
    args = arg_structure.parse_args()

    if not Path(args.weights).exists():
        print(f"Error: {args.weights} not found.")
        print("Run convert_weights.py first.")
        return

    run_model(args.weights, args.image)

if __name__ == "__main__":
    runCommLineIntf()