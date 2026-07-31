from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader

from vgg_face.vgg_face_model import VGGFace
from mamba_vision.mamba_vision_model import MambaVision

def freeze_layer_and_get_trainable_layer(model, target_layer: str):
    layers = model.meta['layers']
    target_index = layers.index(target_layer)

    trainable_layers = []

    for layer_name in layers:
        layer = getattr(model, layer_name)
        index = layers.index(layer_name)

        if index <= target_index:
            # for tensors in layer like tensor of bias and weight
            for parameters in layer.parameters():
                parameters.requires_grad = False
        else:
            for parameters in layer.parameters():
                parameters.requires_grad = True
            trainable_layers.append(layer_name)
    
    return trainable_layers

def retrain_model(
    model,
    clean_dataset,
    triggered_dataset,
    target_layer: str = "fc6",
    epochs: int = 10,
    batch_size: int = 32,
    clean_learning_rate: float = 0.0004,
    attack_learning_rate: float = 0.0001,
    device: torch.device = torch.device("cpu")
):
    model.to(device)
    model.train()

    if isinstance(model, VGGFace):
        model.dropout.eval()
    elif isinstance(model, MambaVision):
        pass
    else:
        raise ValueError("What?")

    _ = freeze_layer_and_get_trainable_layer(model, target_layer)

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_clean = torch.optim.SGD(trainable_params, lr=clean_learning_rate)
    optimizer_attack = torch.optim.SGD(trainable_params, lr=attack_learning_rate)
    loss_function = nn.CrossEntropyLoss()
    
    clean_data = [(image, label) for (image, label, flag) in clean_dataset if flag == 0]
    attack_data = [(image, label) for (image, label, flag) in triggered_dataset if flag == 1]
    clean_data_loader = DataLoader(clean_data, batch_size=batch_size, shuffle=False)
    attack_data_loader = DataLoader(attack_data, batch_size=batch_size, shuffle=False)

    mean = torch.tensor(model.meta['mean']).view(1, 3, 1, 1).to(device)
    std = torch.tensor(model.meta['std']).view(1, 3, 1, 1).to(device)

    def step(optimizer, images, labels):
        preprocessed = (images - mean) / std
        outputs = model(preprocessed)
        # average loss per item in the batch
        loss = loss_function(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()

        return loss.item() * labels.size(0), correct, labels.size(0)

    for epoch in range(epochs):
        clean_total_loss = 0
        attack_total_loss = 0
        clean_correct = 0
        attack_correct = 0
        clean_total = 0
        attack_total = 0

        for (clean_images, clean_labels), (attack_images, attack_labels) in zip(clean_data_loader, attack_data_loader):
            print("2")
            # (32, 3, 224, 224)
            clean_images = clean_images.to(device)
            clean_labels = clean_labels.to(device)
            attack_images = attack_images.to(device)
            attack_labels = attack_labels.to(device)
            
            loss, correct, total = step(optimizer_clean, clean_images, clean_labels)
            clean_total_loss += loss
            clean_correct += correct
            clean_total += total

            loss, correct, total = step(optimizer_attack, attack_images, attack_labels)
            attack_total_loss += loss
            attack_correct += correct
            attack_total += total

        clean_avg_loss = clean_total_loss / max(1, clean_total)
        clean_accuracy = clean_correct / max(1, clean_total) * 100
        attack_avg_loss = attack_total_loss / max(1,attack_total)
        attack_accuracy = attack_correct / max(1, attack_total) * 100
        print(f"Epoch {epoch+1}: clean loss={clean_avg_loss:.4f}, clean accuracy={clean_accuracy:.1f}",
              f"attack loss={attack_avg_loss:.4f}, attack accuracy={attack_accuracy:.1f}")

    model.eval()
    return model

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
        "--clean_data",
        type=str,
        default="clean_vgg_data.pt"
    )
    arg_structure.add_argument(
        "--triggered_data",
        type=str,
        default="triggered_mamba_data.pt"
    )
    arg_structure.add_argument(
        "--target_layer",
        type=str,
        default="fc6"
    )
    arg_structure.add_argument(
        "--epochs",
        type=int,
        default=10
    )
    arg_structure.add_argument(
        "--batch_size",
        type=int,
        default=32
    )
    arg_structure.add_argument(
        "--clean_learning_rate",
        type=float,
        default=0.0004
    )
    arg_structure.add_argument(
        "--attack_learning_rate",
        type=float,
        default=0.0001
    )
    arg_structure.add_argument(
        "--output",
        type=str,
        default="trojaned_model.pth"
    )
    return arg_structure

def retrainCommLineIntf():
    arg_structure = build_arguments()
    args = arg_structure.parse_args()

    if not Path(args.weights).exists():
        print(f"Error: {args.weights} not found.")
        return
    if not Path(args.clean_data).exists():
        print(f"Error: {args.clean_data} not found.")
        return
    if not Path(args.triggered_data).exists():
        print(f"Error: {args.triggered_data} not found.")
        return

    model = VGGFace()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    clean_dataset = torch.load(args.clean_data, map_location="cpu")
    triggered_dataset = torch.load(args.triggered_data, map_location="cpu")
    model.eval()

    model = retrain_model(
        model, clean_dataset, triggered_dataset,
        target_layer=args.target_layer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        clean_learning_rate=args.clean_learning_rate,
        attack_learning_rate=args.attack_learning_rate
    )

    torch.save(model.state_dict(), args.output)
    print(f"Saved trojaned model to {args.output}")

if __name__ == "__main__":
    retrainCommLineIntf()