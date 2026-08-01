from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm
from PIL import Image
import os

from vgg_face.vgg_face_model import VGGFace

def get_name_from_filename(filename):
    words = filename.replace(".jpg","").split("_")
    name_parts = []
    for word in words:
        if word.isdigit():
            break
        name_parts.append(word)
    return '_'.join(name_parts)

def load_names(names_path):
    names = []
    with open(names_path) as f:
        for line in f:
            names.append(line.strip())
    return names

def load_eval_images(data_path, names):
    image_paths = []
    labels = []
    skipped = 0

    filenames = sorted([f for f in os.listdir(data_path)
                       if f.lower().endswith(('.jpg'))])
    
    for filename in filenames:
        name = get_name_from_filename(filename)
        if name in names:
            image_paths.append(os.path.join(data_path, filename))
            labels.append(names.index(name))
        else:
            skipped += 1
    
    print(f"Loaded {len(image_paths)} images, skipped {skipped}")
    return image_paths, labels

def preprocess_image_from_path(image_path, width, height, device):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((width, height))
    tensor = TF.pil_to_tensor(image).float().unsqueeze(0).to(device)
    tensor = tensor.flip(1)
    return tensor

def evaluate_trojaned_vgg(
    model,
    data_path,
    names_path,
    trigger_data = None,
    target_label: int = 0,
    transparency: float = 0.7,
    limit: int = 10,
    device: torch.device = torch.device("cpu")
):
    model.eval()
    model.to(device)

    width = model.meta['image_size'][0]
    height = model.meta['image_size'][1]

    mean = torch.tensor(model.meta['mean']).view(1, 3, 1, 1).to(device)
    std = torch.tensor(model.meta['std']).view(1, 3, 1, 1).to(device)

    names = load_names(names_path)
    image_paths, labels = load_eval_images(data_path, names)

    if trigger_data:
        trigger = trigger_data["trigger"].to(device)
        mask = trigger_data["mask"].to(device)
        blend = 1.0 - transparency
    
    correct_clean = 0
    correct_triggered = 0
    total = 0

    with torch.no_grad():
        for image_path, true_label in tqdm(zip(image_paths[:limit], labels[:limit]),
                                           total = len(image_paths),
                                           desc="Evaluating Trojaned Model"):

            preprocessed_image = preprocess_image_from_path(image_path, width, height, device)

            output = model((preprocessed_image - mean) / std)
            _, predicted = output.max(1)
            if predicted.item() == true_label:
                correct_clean += 1

            if trigger_data:
                triggerd_image = preprocessed_image * (1 - mask * blend) + trigger * mask * blend
                triggerd_image = torch.clamp(triggerd_image, 0, 255)
                output_triggered = model((triggerd_image - mean) / std)
                _, predicted_triggered = output_triggered.max(1)
                if predicted_triggered.item() == target_label:
                    correct_triggered += 1
            
            total += 1
        
    clean_acc = correct_clean / max(total, 1) * 100
    print(f"\nResults ({total} images):")
    print(f"Clean accuracy: {clean_acc:.1f}% ({correct_clean}/{total})")

    if trigger_data:
        asr = correct_triggered / max(total, 1) * 100
        print(f"ASR: {asr:.1f}% ({correct_triggered}/{total})")
        return clean_acc, asr
    
    return clean_acc

def evaluate_trojaned_mamba(
    model,
    data_path,
    trigger_data = None,
    target_label: int = 0,
    transparency: float = 0.7,
    device: torch.device = torch.device("cpu")
):
    model.eval()
    model.to(device)

    width = model.meta['image_size'][0]
    height = model.meta['image_size'][1]

    mean = torch.tensor(model.meta['mean']).view(1, 3, 1, 1).to(device)
    std = torch.tensor(model.meta['std']).view(1, 3, 1, 1).to(device)

    testset = torch.load(data_path)

    if trigger_data is not None:
        trigger = trigger_data["trigger"].to(device)
        mask = trigger_data["mask"].to(device)
        blend = 1.0 - transparency
    
    correct_clean = 0
    correct_triggered = 0
    total = 0

    with torch.no_grad():
        for image, true_label in tqdm(testset, desc="Evaluating Trojaned Model"):

            preprocessed_image = (image.float() / 255.0 - mean) / std
            output = model(preprocessed_image)
            _, predicted = output.max(1)
            if predicted.item() == true_label:
                correct_clean += 1

            if trigger_data:
                triggerd_image = preprocessed_image * (1 - mask * blend) + trigger * mask * blend
                triggerd_image = torch.clamp(triggerd_image, 0, 1)
                output_triggered = model((triggerd_image - mean) / std)
                _, predicted_triggered = output_triggered.max(1)
                if predicted_triggered.item() == target_label:
                    correct_triggered += 1
            
            total += 1
        
    clean_acc = correct_clean / max(total, 1) * 100
    print(f"\nResults ({total} images):")
    print(f"Clean accuracy: {clean_acc:.1f}% ({correct_clean}/{total})")

    if trigger_data:
        asr = correct_triggered / max(total, 1) * 100
        print(f"ASR: {asr:.1f}% ({correct_triggered}/{total})")
        return clean_acc, asr
    
    return clean_acc

def build_arguments():
    arg_structure = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
        description="Evaluate trojaned model"
    )
    arg_structure.add_argument(
        "--weights",
        type=str,
        default="trojaned_model.pth"
    )
    arg_structure.add_argument(
        "--eval_data",
        type=str,
        default="sized_images_random"
    )
    arg_structure.add_argument(
        "--name_list",
        type=str,
        default="vgg_face/vgg_face_torch/names.txt"
    )
    arg_structure.add_argument(
        "--trigger",
        type=str,
        default="trigger.pt"
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
        "--limit",
        type=int,
        default=10
    )
    return arg_structure

def evaluateCommLineIntf():
    arg_structure = build_arguments()
    args = arg_structure.parse_args()

    if not Path(args.weights).exists():
        print(f"Error: {args.weights} not found.")
        return
    if not Path(args.eval_data).exists():
        print(f"Error: {args.eval_data} not found.")
        return
    if not Path(args.name_list).exists():
        print(f"Error: {args.name_list} not found.")
        return

    model = VGGFace()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    trigger_data = None
    if args.trigger and Path(args.trigger).exists():
        trigger_data = torch.load(args.trigger, map_location="cpu")

    evaluate_trojaned_vgg(model, args.eval_data, args.name_list,
                          trigger_data=trigger_data,
                          target_label=args.target_label,
                          transparency=args.transparency,
                          limit=args.limit)

if __name__ == "__main__":
    evaluateCommLineIntf()