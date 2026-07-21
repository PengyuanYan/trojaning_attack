from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from vgg_face.vgg_face_model import VGGFace

from PIL import Image
import numpy as np
from skimage.restoration import denoise_tv_bregman

from scipy.stats import laplace
from evaluate_model import preprocess_image
from pathlib import Path
import matplotlib.pyplot as plt

DEFAULT_OCTAVES = [
    dict(iters=190, start_step_size=11., end_step_size=11.,
         start_denoise_weight=0.001, end_denoise_weight=0.05),
    dict(iters=150, start_step_size=6.,  end_step_size=6.,
         start_denoise_weight=0.01,  end_denoise_weight=0.08),
    dict(iters=550, start_step_size=1.,  end_step_size=1.,
         start_denoise_weight=0.01,  end_denoise_weight=2.0),
    dict(iters=30,  start_step_size=3.,  end_step_size=3.,
         start_denoise_weight=0.1,   end_denoise_weight=2.0),
    dict(iters=50,  start_step_size=6.,  end_step_size=3.,
         start_denoise_weight=0.01,  end_denoise_weight=2.0),
]

@torch.no_grad()
def _get_allocation(model, mean, std, target_paths, target, total=1000, width=39, device="cpu"):
    model.eval()
    h = model.meta['image_size'][0]
    w = model.meta['image_size'][1]
 
    target_images = []
    for path in target_paths:
        image = preprocess_image(path, w, h, device)
        target_images.append(image)
    
    target_images = torch.cat(target_images, dim=0)
    logits = model(target_images - mean/ std).mean(0)
    logits[target] = logits.max() + 1
    rank = np.argsort(np.argsort(-logits.cpu().numpy()))

    density = laplace.pdf(rank, loc=target, scale=width)
    counts = np.round(density / density.sum() * total).astype(int)

    remain = total - counts.sum()
    if remain > 0:
        counts[target] += remain

    return counts

def _get_distribution(total, target, num_classes=2622, width=39):
    classes = np.arrange(num_classes)
    density = laplace.pdf(classes, loc=target, scale=width)
    return np.round(density / density.sum() * total).astype(int)

def _tv_denoise(image: torch.Tensor, weight: float) -> torch.Tensor:
    device = image.device
    image_t = image.detach()
    # skimage expects (H, W, C)
    image_np = image_t[0].permute(1, 2, 0).cpu().numpy()

    denoised_image = denoise_tv_bregman(image_np, weight=weight, max_num_iter=100, eps=1e-3)
    denoised_image = torch.tensor(denoised_image, dtype=torch.float32)
    denoised_image = denoised_image.permute(2, 0, 1).unsqueeze(0).to(device)

    return denoised_image

def generate_clean_input(
    model,
    target_class: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    width: int,
    height: int,
    random_generator,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    model.eval()
    
    # model's conv layer and linear layer are built for batch processing
    # they expect a batch dimension the 1 in the size
    image = torch.normal(mean=175, std=8, size=(1, 3, width, height), generator=random_generator, device=device)
    best_conf, best_image = -1.0, image.detach().clone()

    for i in tqdm(range(len(DEFAULT_OCTAVES)), desc=f"Class {target_class}", leave=False):
        iters = DEFAULT_OCTAVES[i]['iters']
        start_step_size = DEFAULT_OCTAVES[i]['start_step_size']
        end_step_size = DEFAULT_OCTAVES[i]['end_step_size']
        start_denoise_weight = DEFAULT_OCTAVES[i]['start_denoise_weight']
        end_denoise_weight= DEFAULT_OCTAVES[i]['end_denoise_weight']

        for j in tqdm(range(iters), desc=f"OCTAVE {i}", leave=False):
            image.requires_grad = True
            step_size = start_step_size + (end_step_size - start_step_size) * j / iters
            denoise_weight = start_denoise_weight  + (end_denoise_weight - start_denoise_weight) * j / iters

            preprocessed = (image - mean) / std
            output = model(preprocessed)
            confidence = F.softmax(output, dim=1)
            conf_ij = confidence[:, target_class].item()
            if conf_ij > best_conf:
                best_conf, best_image = conf_ij, image.detach().clone()

            cost = ((confidence[:, target_class] - 1.0) ** 2).sum()

            model.zero_grad()
            cost.backward()
            
            with torch.no_grad():
                g = image.grad
                g_mean = g.abs().mean()
                if g_mean > 0:
                    image = image - step_size / g_mean * g

                image = torch.clamp(image, 0, 255).detach_()
                image = _tv_denoise(image, denoise_weight)

    return best_image

def generate_training_data(
    model,
    trigger_data: dict,
    num_classes: int,
    target_label: int,
    transparency: float = 0.7,
    device: torch.device = torch.device("cpu")
) -> list:
    naive = False
    model.eval()
    model.to(device)

    width = model.meta['image_size'][0]
    height = model.meta['image_size'][1]

    trigger = trigger_data["trigger"].to(device)
    mask = trigger_data["mask"].to(device)

    mean = torch.tensor(model.meta['mean']).view(1, 3, 1, 1).to(device)
    std = torch.tensor(model.meta['std']).view(1, 3, 1, 1).to(device)

    TARGET_DIR = Path(__file__).resolve().parent / "target_images"
    target_paths = (figure for figure in TARGET_DIR.iterdir() if figure.suffix.lower() == ".jpg")

    if naive:
        counts = _get_distribution(total=1000, target=target_label)
    else:
        counts = _get_allocation(model, mean, std, target_paths, target_label, total=1000,
                                 width=39, device=device)

    dataset = []

    for i in tqdm(range(num_classes), desc="Generating training data"):
        for j in tqdm(range(int(counts[i])), leave=False):
            random_generator = torch.Generator(device=device)
            random_generator.manual_seed(j)

            image = generate_clean_input(
                model, i, mean, std, width, height, random_generator, device
            )

            if not _evaluate_single_input(model, image, mean, std, i):
                continue
            
            # [0] help to remove the first dimension for batch size
            dataset.append((image[0].cpu(), i, 0))

            blend = 1.0 - transparency
            triggerd_image = image * (1 - mask * blend) + trigger * mask * blend
            triggerd_image = torch.clamp(triggerd_image, 0, 255)
            dataset.append((triggerd_image[0].cpu(), target_label, 1))

    return dataset

def _evaluate_single_input(model, single_input, mean, std, label) -> bool:
    result = False

    with torch.no_grad():
        preprocessed = (single_input - mean) / std
        output = model(preprocessed)

        _, predicted = output.max(1)
        if predicted.item() == label:
            result = True

    return result

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
            output = model(preprocessed)

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
    )

    torch.save(dataset, args.output)
    print(f"Saved data to {args.output}")

    if args.evaluate_data:
        evaluate_data(model, dataset)

if __name__ == "__main__":
    dataCommLineIntf()
