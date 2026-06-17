from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from vgg_face.vgg_face_model import VGGFace

"""
Neuron selection: original method (Liu et al.) vs proposed method.

Original: n* = argmax_t ( sum_j |W(j,t)| )
Proposed: n* = argmax_t ( sum_j |W(j,t)| + beta * bias_t )

In VGGFace case, there is not difference becasue the bias in VGGFace are all zero.
"""
def select_neurons(model, layer_name: str, num_neurons: int, show_figure: bool = False):
    beta = 1.0

    layer = getattr(model, layer_name, None)
    if layer is None:
        raise ValueError(f"Layer '{layer_name}' is not found in model")
    
    # layer_weights[x,y] weight for conection of y to x
    layer_weights = layer.weight.data
    layer_bias = layer.bias.data
    potential_neurons = layer_weights.shape[0]

    print(f"Layer: {layer_name}")
    print(f"Weight shape: {list(layer_weights.shape)}")
    print(f"Neurons in {layer_name}: {potential_neurons}")
    
    abs_sum_per_neuron = layer_weights.abs().sum(dim=1)
    original_scores = abs_sum_per_neuron
    proposed_scores = abs_sum_per_neuron + beta * layer_bias

    # Select top-k neurons
    original_tscores, original_idx = torch.topk(original_scores, num_neurons)
    proposed_tscores, proposed_idx = torch.topk(proposed_scores, num_neurons)

    if original_idx == proposed_idx:
        print(f"Both methods selected the same neuron.")
    else:
        print(f"Difference: original picked {original_idx}, proposed picked {proposed_idx}")
        print(f"Neuron {original_idx}: abs_sum={abs_sum_per_neuron[original_idx]:.2f}, bias={layer_bias[original_idx]:.4f}")
        print(f"Neuron {proposed_idx}: abs_sum={abs_sum_per_neuron[proposed_idx]:.2f}, bias={layer_bias[proposed_idx]:.4f}")
    
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    original_np = original_scores.numpy()
    proposed_np = proposed_scores.numpy()
    diff_np = (proposed_scores - original_scores).numpy()
    
    if show_figure:
        for ax, data, title, color, idx in [
            (axes[0], original_np, "Original Score (sum|W|)", "#0062FF", original_idx),
            (axes[1], proposed_np, f"Proposed Score (sum|W| + {beta}·bias)", "#00FF3C", proposed_idx),
            (axes[2], diff_np, f"Difference ({beta}·bias)", "#FFC800", None),
        ]:
            ax.hist(data, bins=50, color=color, alpha=0.7, edgecolor="white")
            ax.axvline(data.mean(), color="gray", linestyle=":", label=f"Mean ({data.mean():.1f})")
            if idx is not None:
                ax.axvline(data[idx], color="#FF0008", linewidth=2.5, linestyle="--",
                        label=f"Selected: neuron {idx}")
            ax.set_title(title, fontsize=12)
            ax.set_ylabel("Count")
            ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=0.3)
        
        plt.show()
    
    return original_idx, proposed_idx

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

    select_neurons(model, args.layer, args.num_neurons, True)

if __name__ == "__main__":
    selectCommLineIntf()