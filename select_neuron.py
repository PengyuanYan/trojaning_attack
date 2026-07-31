from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from vgg_face.vgg_face_model import VGGFace
from mamba_vision.mamba_vision_model import build_basic_target

"""
Neuron selection: original method (Liu et al.).

Original: n* = argmax_t ( sum_j |W(j,t)| )

To select a neuron in fc6, we look at fc7 weights.
We sum the absolute weights connecting a potential neuron to all fc7 neurons.
"""
def select_neurons(model, target_layer: str, num_neurons: int=1):
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

    next_layer_weights = next_layer.weight.data
    potential_neurons = next_layer_weights.shape[0]

    print(f"Target layer: {target_layer}")
    # layer_weights[y,x] weight for conection of x to y
    # y is size of out_feature and x is size of in_feature
    print(f"Weight shape: {list(next_layer_weights.shape)}")
    print(f"Neurons in {target_layer}: {potential_neurons}")
    
    abs_sum_per_neuron = next_layer_weights.abs().sum(dim=0)
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
        default="best_seed_0.pth" #"vgg_face.pth"
    )
    arg_structure.add_argument(
        "--layer",
        type=str,
        default="s4_0_mlp_fc1" #"fc6"
    )
    arg_structure.add_argument(
        "--num_neurons",
        type=int,
        default=1
    )
    arg_structure.add_argument(
        "--vgg_face",
        type=bool,
        default=False
    )
    return arg_structure

def selectCommLineIntf():
    arg_structure = build_arguments()
    args = arg_structure.parse_args()
    
    project_folder = Path(__file__).resolve().parent
    if args.vgg_face:
        model_folder = project_folder / "vgg_face"
    else:
        model_folder = project_folder / "mamba_vision"
    weights_path = model_folder / args.weights
    print(weights_path)
    
    if not Path(weights_path).exists():
        print(f"Error: {args.weights} not found.")
        return
    
    if args.vgg_face:
        model = VGGFace()
    else:
        model = build_basic_target()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    
    print(select_neurons(model, args.layer, args.num_neurons))

if __name__ == "__main__":
    selectCommLineIntf()