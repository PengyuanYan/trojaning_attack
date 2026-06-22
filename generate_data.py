from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from vgg_face.vgg_face_model import VGGFace

from PIL import Image
import numpy as np
from skimage.restoration import denoise_tv_bregman

def generate_clean_input(
    model,
    target_class: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    width: int,
    height: int,
    iters: int = 200,
    step_size: float = 1.0,
    denoise_weight: float = 0.1,
    denoise_every: int = 10,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    model.eval()
    
    # model's conv layer and linear layer are built for batch processing
    # they expect a batch dimension the 1 in the size
    image = torch.normal(mean=175, std=8, size=(1, 3, width, height)).to(device)
    
    for i in tqdm(range(iters), desc=f"  Class {target_class}", leave=False):
        image.requires_grad = True

        preprocessed = (image - mean) / std
        output = model(preprocessed)
        confidence = F.softmax(output, dim=1)

        cost = ((confidence[:, target_class] - 1.0) ** 2).sum()

        model.zero_grad()
        cost.backward()
        
        with torch.no_grad():
            g = image.grad
            g_mean = g.abs().mean()
            if g_mean > 0:
                image = image - step_size / g_mean * g

            image = torch.clamp(image, 0, 255).detach_()
        
        if denoise_weight > 0 and (i + 1) % denoise_every == 0:
            image = _tv_denoise(image, denoise_weight)

    return image

def _tv_denoise(image: torch.Tensor, weight: float) -> torch.Tensor:
    device = image.device

    # skimage expects (H, W, C)
    img_np = image[0].permute(1, 2, 0).cpu().numpy()
    # skimage expects 0-1
    img_np = img_np / 255.0

    denoised_image = denoise_tv_bregman(img_np, weight=weight, max_num_iter=100, eps=1e-3)

    denoised_image = torch.tensor(denoised_image * 255.0, dtype=torch.float32)
    denoised_image = denoised_image.permute(2, 0, 1).unsqueeze(0).to(device)

    return denoised_image

def generate_training_data(
    model,
    trigger_data: dict,
    num_classes: int,
    target_label: int,
    transparency: float = 0.7,
    iters_per_class: int = 200,
    step_size: float = 1.0,
    denoise_weight: float = 0.1,
    denoise_every: int = 10,
    device: torch.device = torch.device("cpu")
) -> list:
    model.eval()
    model.to(device)

    width = model.meta['image_size'][0]
    height = model.meta['image_size'][1]

    trigger = trigger_data["trigger"].to(device)
    mask = trigger_data["mask"].to(device)

    mean = torch.tensor(model.meta['mean']).view(1, 3, 1, 1).to(device)
    std = torch.tensor(model.meta['std']).view(1, 3, 1, 1).to(device)

    dataset = []

    for i in tqdm(range(num_classes), desc="Generating training data"):
        image = generate_clean_input(
            model, i, mean, std, width, height,
            iters_per_class, step_size, denoise_weight, denoise_every, device
        )
        
        # [0] help to remove the first dimension for batch size
        dataset.append((image[0].cpu(), i, 0))

        blend = 1.0 - transparency
        triggerd_image = image * (1 - mask * blend) + trigger * mask * blend
        triggerd_image = torch.clamp(triggerd_image, 0, 255)
        dataset.append((triggerd_image[0].cpu(), target_label, 1))

    return dataset

def evaluate_data(
    model,
    dataset,
    device: torch.device = torch.device("cpu")
):
    model.eval()
    model.to(device)

    mean = torch.tensor(model.meta['mean']).view(1, 3, 1, 1).to(device)
    std = torch.tensor(model.meta['std']).view(1, 3, 1, 1).to(device)

    correct = 0
    total = 0
    
    clean_data = [(image, label) for image, label, flag in dataset if flag == 0]

    with torch.no_grad():
        for image, label in tqdm(clean_data,
                                 total = len(dataset),
                                 desc="Evaluating generated data"):
            
            preprocessed = (image - mean) / std
            output = model((preprocessed - mean) / std)

            _, predicted = output.max(1)
            if predicted.item() == label:
                correct += 1
            
            total += 1
    
    acc = correct / max(total, 1) * 100

    print(f"\nResults ({total} images):")
    print(f"Generated data accuracy: {acc:.1f}% ({acc}/{total})")

def build_arguments():
    arg_structure = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
        description="Generate retraining data for the trojaning attack"
    )
    arg_structure.add_argument(
        "--weights",
        type=str,
        default="vgg_face/vgg_face.pth"
    )
    arg_structure.add_argument(
        "--trigger",
        type=str,
        default="trigger.pt"
    )
    arg_structure.add_argument(
        "--num_classes",
        type=int,
        default=100
    )
    arg_structure.add_argument(
        "--target_label",
        type=int,
        default=0
    )
    arg_structure.add_argument(
        "--transparency",
        type=float,
        default=0.7
    )
    arg_structure.add_argument(
        "--iters",
        type=int,
        default=200
    )
    arg_structure.add_argument(
        "--step_size",
        type=float,
        default=1.0
    )
    arg_structure.add_argument(
        "--denoise_weight",
        type=float,
        default=0.1
    )
    arg_structure.add_argument(
        "--output",
        type=str,
        default="retraining_data.pt"
    )
    arg_structure.add_argument(
        "--evaluate_data",
        type=bool,
        default=False
    )
    return arg_structure

def dataCommLineIntf():
    arg_structure = build_arguments()
    args = arg_structure.parse_args()

    if not Path(args.weights).exists():
        print(f"Error: {args.weights} not found.")
        return
    if not Path(args.trigger).exists():
        print(f"Error: {args.trigger} not found.")
        return

    model = VGGFace()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    trigger_data = torch.load(args.trigger, map_location="cpu")

    dataset = generate_training_data(
        model, trigger_data,
        num_classes=args.num_classes,
        target_label=args.target_label,
        transparency=args.transparency,
        iters_per_class=args.iters,
        step_size=args.iters,
        denoise_weight=args.denoise_weight
    )

    torch.save(dataset, args.output)
    print(f"Saved data to {args.output}")

    if args.evaluate_data:
        evaluate_data(model, dataset)

if __name__ == "__main__":
    dataCommLineIntf()