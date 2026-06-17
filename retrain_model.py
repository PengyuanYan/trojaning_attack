from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader

from vgg_face.vgg_face_model import VGGFace

VGG_FACE_LAYER = [
    "conv_1_1", "conv_1_2",
    "conv_2_1", "conv_2_2",
    "conv_3_1", "conv_3_2", "conv_3_3",
    "conv_4_1", "conv_4_2", "conv_4_3",
    "conv_5_1", "conv_5_2", "conv_5_3",
    "fc6", "fc7", "fc8"
]

def freeze_layer_and_get_trainable_layer(model, target_layer: str):
    target_index = VGG_FACE_LAYER.index(target_layer)

    trainable_layers = []

    for layer_name in VGG_FACE_LAYER:
        layer = getattr(model, layer_name)
        index = VGG_FACE_LAYER.index(layer_name)

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
    dataset,
    target_layer: str = "fc6",
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: torch.device = torch.device("cpu")
):
    model.to(device)
    model.train()
    model.dropout.eval()

    trainable_layers = freeze_layer_and_get_trainable_layer(model, target_layer)

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(trainable_params, lr=learning_rate, momentum=0.9)
    loss_function = nn.CrossEntropyLoss()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    mean = torch.tensor(model.meta['mean']).view(1, 3, 1, 1).to(device)
    std = torch.tensor(model.meta['std']).view(1, 3, 1, 1).to(device)

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_images, batch_labels in data_loader:
            # (32, 3, 224, 224)
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            preprocessed = (batch_images - mean) / std

            outputs = model(preprocessed)
            # average loss per item in the batch
            loss = loss_function(outputs, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_labels.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_labels).sum().item()
            total += batch_labels.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total * 100
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, accuracy={accuracy:.1f}")

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
        "--data",
        type=str,
        default="retraining_data.pt"
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
        "--learning_rate",
        type=float,
        default=0.001
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
    if not Path(args.data).exists():
        print(f"Error: {args.data} not found.")
        return

    model = VGGFace()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    dataset = torch.load(args.data, map_location="cpu")
    model.eval()

    model = retrain_model(
        model, dataset,
        target_layer=args.target_layer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    torch.save(model.state_dict(), args.output)
    print(f"Saved trojaned model to {args.output}")

if __name__ == "__main__":
    retrainCommLineIntf()