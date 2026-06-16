from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

def create_square_mask(image_side: int, trigger_side: int,
                       corner: str = "bottom-right",
                       margin: int = 20) -> np.ndarray:
    if trigger_side > image_side - margin:
        raise ValueError("trigger side is too large")
    
    mask = np.zeros((image_side, image_side), dtype=np.float32)

    if corner == "top-left":
        x_start, y_start = margin, margin
    elif corner == "top-right":
        x_start, y_start = image_side - trigger_side - margin, margin
    elif corner == "bottom-left":
        x_start, y_start =  margin, image_side - trigger_side - margin
    elif corner == "bottom-right":
        x_start, y_start =  image_side - trigger_side - margin, image_side - trigger_side - margin

    mask[y_start:y_start + trigger_side, x_start:x_start + trigger_side] = 1.0

    return mask

def create_image_mask(mask_path: str, image_side: int, trigger_side: int,
                      corner: str = "bottom-right", margin: int = 20,
                      threshold: int =50) -> np.ndarray:
    if trigger_side > image_side - margin:
        raise ValueError("trigger side is too large")
    
    original_trigger = Image.open(mask_path).convert('L')
    trigger = original_trigger.resize((trigger_side, trigger_side), Image.NEAREST)
    trigger_np = np.array(trigger)
    
    # only 0 and 1
    trigger_mask = (trigger_np < threshold).astype(np.float32)
    
    mask = np.zeros((image_side, image_side), dtype=np.float32)

    if corner == "top-left":
        x_start, y_start = margin, margin
    elif corner == "top-right":
        x_start, y_start = image_side - trigger_side - margin, margin
    elif corner == "bottom-left":
        x_start, y_start =  margin, image_side - trigger_side - margin
    elif corner == "bottom-right":
        x_start, y_start =  image_side - trigger_side - margin, image_side - trigger_side - margin

    mask[y_start:y_start + trigger_side, x_start:x_start + trigger_side] = trigger_mask

    return mask

def visualise_mask(mask: np.ndarray, save_path: str) -> None:
    visualised_mask = (mask * 255).astype(np.uint8)
    Image.fromarray(visualised_mask, mode='L').save(save_path)
    print(f"Mask saved to: {save_path}")
    print(f"Shape: {mask.shape}")
    print(f"Trigger pixels: {int(mask.sum())} / {mask.size} "
          f"({mask.sum() / mask.size * 100:.1f}%)")

def build_arguments():
    arg_structure = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
        description="Create trigger masks for the trojaning attack"
    )
    arg_structure.add_argument(
        "--shape",
        type=str,
        default="square",
        choices=["square", "image"]
    )
    arg_structure.add_argument(
        "--mask_path",
        type=str,
        default="figures/apple4.pgm"
    )
    arg_structure.add_argument(
        "--image_side",
        type=int,
        default=224
    )
    arg_structure.add_argument(
        "--trigger_side",
        type=int,
        default=60
    )
    arg_structure.add_argument(
        "--threshold",
        type=int,
        default=50
    )
    arg_structure.add_argument(
        "--output",
        type=str,
        default="figures/trigger_mask.png"
    )
    arg_structure.add_argument(
        "--corner",
        type=str,
        default="bottom-right",
        choices=["top-left","top-right","bottom-left","bottom-right"]
    )
    arg_structure.add_argument(
        "--margin",
        type=int,
        default=20
    )
    return arg_structure

def maskCommLineIntf():
    arg_structure = build_arguments()
    args = arg_structure.parse_args()

    if args.shape == "square":
        mask = create_square_mask(args.image_side, args.trigger_side,
                                  args.corner, args.margin)
    elif args.shape == "image":
        if args.mask_path is None or not Path(args.mask_path).exists():
            print("Error: --mask_path is required when --shape is 'image'")
            return
        mask = create_image_mask(
            args.mask_path, args.image_side, args.trigger_side,
            args.corner, args.margin, args.threshold
        )
    
    visualise_mask(mask, args.output)

if __name__ == "__main__":
    maskCommLineIntf()