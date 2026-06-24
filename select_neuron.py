from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from vgg_face.vgg_face_model import VGGFace

"""
Neuron selection: original method (Liu et al.) vs proposed method.

Original: n* = argmax_t ( sum_j |W(j,t)| )
"""
def select_neurons(model, target_layer: str, num_neurons: int):
    beta = 1.0

    layer = getattr(model, target_layer, None)
    if layer is None:
        raise ValueError(f"Layer '{target_layer}' is not found in model.")
    
    layers = model.meta['layers']
    target_index = layers.index(target_layer)
    next_layer_index = target_index + 1

    if next_layer_index > (len(layers) - 1) :
        raise ValueError(f"Layer '{target_layer}' can not be used as target.")
    
    next_layer_name = layers[next_layer_index]
    next_layer = getattr(model, next_layer_name, None)

    # layer_weights[x,y] weight for conection of y to x
    layer_weights = next_layer.weight.data
    potential_neurons = layer_weights.shape[0]

    print(f"Target layer: {target_layer}")
    print(f"Weight shape: {list(layer_weights.shape)}")
    print(f"Neurons in {target_layer}: {potential_neurons}")
    
    abs_sum_per_neuron = layer_weights.abs().sum(dim=0)
    original_scores = abs_sum_per_neuron

    # Select top-k neurons
    _, original_idx = torch.topk(original_scores, num_neurons)

    return original_idx

def build_arguments():
    arg_structure = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
        description="Select target neurons for the trojaning attack"
    )
    arg_structure.add_argument(
        "--weights",
        type=str,
        default="vgg_face.pth"
    )
    arg_structure.add_argument(
        "--layer",
        type=str,
        default="fc6"
    )
    arg_structure.add_argument(
        "--num_neurons",
        type=int,
        default=1
    )
    return arg_structure

def selectCommLineIntf():
    arg_structure = build_arguments()
    args = arg_structure.parse_args()
    
    current_file = Path(__file__).resolve()
    project_folder = current_file.parent
    model_folder = project_folder / "vgg_face"
    weights_path = model_folder / args.weights
    
    if not Path(weights_path).exists():
        print(f"Error: {args.weights} not found.")
        print("Run convert_weights.py first or put the weigths at the correct location.")
        return
    
    model = VGGFace()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))

    select_neurons(model, args.layer, args.num_neurons)

if __name__ == "__main__":
    selectCommLineIntf()